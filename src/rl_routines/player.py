""" Here are the functions for the async player.
"""

import time
import torch
from rl_logger import Logger
from wintermute.env_wrappers import get_wrapped_atari
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon
from src.utils import get_process_memory
from src.rl_routines.common import get_policy_evaluation


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

    state, reward, done = env.reset(), 0, False
    ep_cnt = 0

    # Warm up

    start = time.time()

    thompson_sampling = False
    if hasattr(opt, "boot") and opt.boot.k > 1:
        thompson_sampling = opt.boot.is_thompson
    if thompson_sampling:
        policy_evaluation.policy.sample_posterior_idx()

    sent = []
    for step in range(opt.learn_start):
        with torch.no_grad():
            pi = policy_evaluation(state)
        _state, _action = state, pi.action
        state, reward, done, _ = env.step(pi.action)
        transition = (_state, pi, reward, state, done)
        experience_queue.put(transition)
        sent.append(transition)
        play_log.update(
            ep_cnt=int(done),
            rw_per_ep=(reward, int(done)),
            rw_per_step=reward,
            max_q=pi.q_value if pi.q_value else float("-inf"),
            playing_fps=1,
        )
        if done:
            state, reward, done = env.reset(), 0, False
            ep_cnt += 1
            if thompson_sampling:
                policy_evaluation.policy.sample_posterior_idx()
        if not sync_queue.empty() or len(sent) > 1000:
            sync_queue.get()
            del sent[:100]

    while sent:
        sync_queue.get()
        del sent[:100]

    used_ram, used_gpu = get_process_memory()
    play_log.update(ram=used_ram, gpu=used_gpu)
    opt.log.log(play_log, step)
    play_log.reset()

    end = time.time()

    opt.log.log_info(play_log, f"Warm up ends after {end-start:.2f}s. Learn!")

    sent1 = []  # pylint: disable=unused-variable
    for step in range(opt.learn_start + 1, opt.step_no + 1):
        with torch.no_grad():
            pi = opt.policy_evaluation(state)
        _state, _action = state, pi.action
        state, reward, done, _ = env.step(pi.action)
        transition = (_state, pi, reward, state, done)
        experience_queue.put(transition)
        sent.append(transition)

        if step % opt.update_freq == 0:
            sync_queue.get()  # The network won't change until I send "train"
            sent1, sent = sent, []  # Keep only the last four
            experience_queue.put("train")

        # do some logging
        play_log.update(
            ep_cnt=int(done),
            rw_per_ep=(reward, int(done)),
            rw_per_step=reward,
            max_q=pi.q_value if pi.q_value else float("-inf"),
            playing_fps=1,
        )

        if done:
            state, reward, done = env.reset(), 0, False
            ep_cnt += 1
            if thompson_sampling:
                policy_evaluation.policy.sample_posterior_idx()
            if ep_cnt % opt.log_freq == 0:
                used_ram, used_gpu = get_process_memory()
                play_log.update(ram=used_ram, gpu=used_gpu)
                opt.log.log(play_log, step)
                play_log.reset()


def init_player(opt, experience_queue, sync_queue):
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
        console_options=("white", "on_green", ["bold"]),
    )
    env = get_wrapped_atari(
        opt.game,
        mode="training",
        seed=opt.seed,
        no_gym=opt.no_gym,
        device=torch.device("cuda"),
    )

    policy_evaluation = get_policy_evaluation(
        opt, opt.estimator, env.action_space.n
    )

    opt.log = log
    opt.env = env
    opt.policy_evaluation = policy_evaluation
    opt.experience_queue = experience_queue
    opt.sync_queue = sync_queue

    play(opt)
