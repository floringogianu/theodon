""" Here are the functions for the async player.
"""

from copy import deepcopy
import torch
from rl_logger import Logger
from wintermute.env_wrappers import get_wrapped_atari
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon
from src.utils import get_process_memory


def play(opt):
    """ Playing loop. Sends transitions to the learner. Sends tran and eval
        signals.
    """
    env = opt.env
    play_log = opt.log.groups["playing"]
    play_log.reset()
    policy_evaluation = opt.policy_evaluation
    experience_queue = opt.experience_queue
    sync_queue = opt.sync_queue
    eval_queue = opt.eval_queue

    state, reward, done = env.reset(), 0, False
    warmed_up = False
    training = False
    ep_cnt = 0

    assert opt.learn_start % 100 == 0, "100 steps used in sync"

    sent1, sent2 = [], []
    last_state = None
    for step in range(1, opt.step_no + 1):

        # take action and save the s to _s and a to _a to be used later
        with torch.no_grad():
            pi = opt.policy_evaluation(state)
        _state, _action = state, pi.action
        state, reward, done, _ = env.step(pi.action)
        transition = (_state, _action, reward, state, done)
        experience_queue.put(transition)

        do_training = (step % opt.update_freq == 0) and warmed_up

        if do_training:
            if training:
                while sent1:
                    sync_queue.get()
                    del sent1[:100]
                sync_queue.get()
                sent2.clear()
            experience_queue.put("train")
            training = True
            sent2.append(transition)
        elif warmed_up:
            sent2.append(transition)
        else:
            sent1.append(transition)
            if not sync_queue.empty():
                sync_queue.get()
                del sent1[:100]

        if step % opt.target_update == 0 and warmed_up:
            experience_queue.put("update_target")

        # do some logging
        play_log.update(
            ep_cnt=int(done),
            rw_per_ep=(reward, int(done)),
            rw_per_step=reward,
            max_q=pi.q_value,
            playing_fps=1,
        )

        if done:
            state, reward, done = env.reset(), 0, False
            ep_cnt += 1

            if ep_cnt % opt.log_freq == 0:
                used_ram, used_gpu = get_process_memory()
                play_log.update(ram=used_ram, gpu=used_gpu)
                opt.log.log(play_log, step)
                play_log.reset()

        warmed_up = step >= opt.learn_start
        if step % opt.eval_freq == 0:
            last_state = deepcopy(
                policy_evaluation.policy.estimator.state_dict()
            )
            eval_queue.put((step, last_state))

    eval_queue.put("done")
    experience_queue.put("done")

    used_ram, used_gpu = get_process_memory()
    play_log.update(ram=used_ram, gpu=used_gpu)
    opt.log.log(play_log, step)
    play_log.reset()


def init_player(opt, experience_queue, sync_queue, eval_queue):
    """ Function to serve as target for the player process.
    """

    log = Logger(label="label", path=opt.out_dir)
    log.add_group(
        tag="playing",
        metrics=(
            log.SumMetric("ep_cnt"),
            log.AvgMetric("rw_per_ep", emph=True),
            log.AvgMetric("rw_per_step"),
            log.MaxMetric("max_q"),
            log.FPSMetric("playing_fps"),
            log.MaxMetric("ram"),
            log.MaxMetric("gpu"),
        ),
        console_options=("white", "on_blue", ["bold"]),
    )
    env = get_wrapped_atari(
        opt.game,
        mode="training",
        seed=opt.seed,
        no_gym=opt.no_gym,
        device=torch.device("cuda"),
    )

    epsilon = get_epsilon(steps=opt.epsilon_steps, end=opt.epsilon_end)
    policy_evaluation = EpsilonGreedyPolicy(
        opt.estimator, env.action_space.n, epsilon
    )

    opt.log = log
    opt.env = env
    opt.policy_evaluation = policy_evaluation
    opt.experience_queue = experience_queue
    opt.sync_queue = sync_queue
    opt.eval_queue = eval_queue

    play(opt)
