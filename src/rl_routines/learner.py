""" Here we define the learner.
"""
from functools import partial
from copy import deepcopy
from torch import optim
from rl_logger import Logger

from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import MemoryEfficientExperienceReplay as ER
from wintermute.replay import PinnedExperienceReplay as PinnedER
from wintermute.replay.prioritized_replay import ProportionalSampler as PER

from src.utils import get_process_memory
from .common import priority_update


def learn(opt):
    """ Here's the learner's loop.
    """

    log = opt.log
    train_log = log.groups["learning"]
    policy_improvement = opt.policy_improvement
    experience_replay = opt.experience_replay
    experience_queue = opt.experience_queue
    sync_queue = opt.sync_queue
    eval_queue = opt.eval_queue
    confirm_queue = opt.confirm_queue

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
        batch = opt.experience_replay.push_and_sample(msg)
        step += opt.update_freq
        msg = experience_queue.get()

        if opt.prioritized:
            policy_improvement(batch, cb=opt.priority_update_cb)
        else:
            policy_improvement(batch)

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

        train_log.update(storage_fps=1)
        train_log.update(training_fps=opt.update_freq)

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
    target_estimator = None

    # construct a policy improvement type
    optimizer = optim.RMSprop(
        estimator.parameters(),
        lr=opt.lr,
        momentum=0.0,
        alpha=0.95,
        eps=0.00001,
        centered=True,
    )
    policy_improvement = DQNPolicyImprovement(
        estimator,
        optimizer,
        gamma=0.99,
        is_double=opt.double,
        target_estimator=target_estimator,
    )

    # we also need an experience replay
    if opt.prioritized:
        experience_replay = PER(
            opt.mem_size,
            batch_size=32,
            alpha=0.6,
            optim_steps=((opt.step_no - opt.learn_start) / opt.update_freq),
        )
        priority_update_cb = partial(priority_update, experience_replay)
        opt.priority_update_cb = priority_update_cb
    else:
        if opt.pinned_memory:
            experience_replay = PinnedER(
                opt.mem_size,
                batch_size=32,
                async_memory=opt.async_memory,
                device=opt.mem_device,
            )
        else:
            experience_replay = ER(
                opt.mem_size, batch_size=32, async_memory=opt.async_memory
            )

    opt.log = log
    opt.policy_improvement = policy_improvement
    opt.experience_replay = experience_replay
    opt.experience_queue = experience_queue
    opt.sync_queue = sync_queue
    opt.eval_queue = eval_queue
    opt.confirm_queue = confirm_queue

    learn(opt)
