""" Here we define the learner.
"""
from functools import partial
from copy import deepcopy
from rl_logger import Logger

from ..utils import get_process_memory
from .common import create_memory, get_policy_improvement


def learn(opt):
    """ Here's the learner's loop
    """

    log = opt.log
    train_log = log.groups["learning"]
    policy_improvement = opt.policy_improvement
    experience_replay = opt.experience_replay
    experience_queue = opt.experience_queue
    sync_queue = opt.sync_queue
    eval_queue = opt.eval_queue
    confirm_queue = opt.confirm_queue

    thompson_sampling = False
    if hasattr(opt, "boot") and opt.boot.k > 1:
        thompson_sampling = opt.boot.is_thompson

    assert opt.learn_start % 100 == 0
    assert opt.learn_start % opt.update_freq == 0
    assert opt.step_no % opt.update_freq == 0
    assert opt.target_update % opt.update_freq == 0

    for step in range(1, opt.learn_start + 1):
        msg = experience_queue.get()
        experience_replay.push(msg)
        train_log.update(storage_fps=1)
        if step % 100 == 0:
            sync_queue.put(1)
    used_ram, used_gpu = get_process_memory()
    train_log.update(ram=used_ram, gpu=used_gpu)
    opt.log.log(train_log, step)
    train_log.reset()

    sync_queue.put(1)  # I'll start

    last_state = None
    step = opt.learn_start
    while step < opt.step_no:
        for _ in range(opt.update_freq - 1):
            msg = experience_queue.get()
            experience_replay.push(msg)
        msg = experience_queue.get()

        if thompson_sampling:
            opt.policy_improvement.set_posterior_idx(msg[1].posterior)        

        batch = opt.experience_replay.push_and_sample(msg)
        step += opt.update_freq
        msg = experience_queue.get()

        if opt.prioritized:  # batch is (data, idxs, weights)
            data, idxs, weights = batch
            update_priorities = partial(
                opt.cb, opt.experience_replay, idxs, weights
            )
            opt.policy_improvement(data, cb=update_priorities)
        else:
            opt.policy_improvement(batch)

        if step % opt.target_update == 0:
            policy_improvement.update_target_estimator()

        if step % opt.eval_freq == 0:
            if last_state:
                confirm_queue.get()
            last_state = (  # TODO: is there a better way?
                deepcopy(policy_improvement.estimator).cpu().state_dict()
            )
            eval_queue.put((step, last_state))

        sync_queue.put(1)

        train_log.update(storage_fps=opt.update_freq)
        train_log.update(training_fps=1)

        if step % opt.learn_log_freq == 0:
            used_ram, used_gpu = get_process_memory()
            train_log.update(ram=used_ram, gpu=used_gpu)
            opt.log.log(train_log, step)
            train_log.reset()

    used_ram, used_gpu = get_process_memory()
    train_log.update(ram=used_ram, gpu=used_gpu)
    opt.log.log(train_log, step)
    train_log.reset()

    if last_state is not None:
        confirm_queue.get()
    eval_queue.put("done")


def init_learner(opt, experience_queue, sync_queue, eval_queue, confirm_queue):
    """ Function to serve as target for the learner process.
    """
    log = Logger(label="label", path=opt.out_dir)
    log.add_group(
        tag="learning",
        metrics=(
            log.FPSMetric("training_fps"),
            log.FPSMetric("storage_fps"),
            log.MaxMetric("ram"),
            log.MaxMetric("gpu"),
        ),
        console_options=("white", "on_blue", ["bold"]),
    )
    estimator = opt.estimator
    policy_improvement = get_policy_improvement(opt, estimator)
    

    # we also need an experience replay
    experience_replay, update_priorities = create_memory(opt)

    opt.log = log
    opt.policy_improvement = policy_improvement
    opt.experience_replay = experience_replay
    opt.experience_queue = experience_queue
    opt.sync_queue = sync_queue
    opt.eval_queue = eval_queue
    opt.confirm_queue = confirm_queue
    opt.cb = update_priorities

    learn(opt)
