""" Entry point.
"""
import time
import random
from functools import partial

import torch
from torch import optim
from rl_logger import Logger

from wintermute.env_wrappers import get_wrapped_atari
from wintermute.estimators import get_estimator
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon

# from wintermute.policy_improvement import get_optimizer
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import NaiveExperienceReplay as ER
from wintermute.replay.prioritized_replay import ProportionalSampler as PER

import liftoff
from liftoff.config import read_config

from src.utils import create_paths


def priority_update(mem, dqn_loss):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss
    mem.update([loss.item() for loss in losses.detach().abs()])
    return (losses * mem.weights.to(losses.device).view_as(losses)).mean()


def train(opt):
    """ Here we do the training.
    """
    env = opt.env
    train_log = opt.log.groups["training"]

    state, reward, done = env.reset(), 0, False
    warmed_up = False
    ep_cnt = 0
    for step in range(1, opt.step_no + 1):

        # take action and save the s to _s and a to _a to be used later
        pi = opt.policy_evaluation(state)
        _state, _action = state, pi.action
        state, reward, done, _ = env.step(pi.action)

        # add a (_s, _a, r, d) transition
        opt.experience_replay.push((_state, _action, reward, state, done))
        # opt.experience_replay.push(_state[0, 3], _action, reward, done)

        # sample a batch and do some learning
        do_training = (step % opt.update_freq == 0) and warmed_up

        if do_training:
            batch = opt.experience_replay.sample()
            if opt.prioritized:
                opt.policy_improvement(batch, cb=opt.priority_update)
            else:
                opt.policy_improvement(batch)

            if step % opt.target_update == 0:
                opt.policy_improvement.update_target_estimator()

        # do some logging
        train_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            rw_per_step=reward,
            max_q=pi.q_value,
            sampling_fps=1,
            training_fps=32 if do_training else 0,
        )

        if done:
            state, reward, done = env.reset(), 0, False
            ep_cnt += 1

            if ep_cnt % opt.log_freq == 0:
                opt.log.log(train_log, step)
                train_log.reset()

        warmed_up = len(opt.experience_replay) > opt.learn_start
    opt.log.log(train_log, step)
    train_log.reset()


def run(opt):
    """ Here we initialize stuff.
    """
    opt.seed = random.randint(0, 1e4) if not opt.seed else opt.seed
    print(f"torch manual seed={opt.seed}.")
    torch.manual_seed(opt.seed)

    # wrap the gym env
    env = get_wrapped_atari(
        f"{opt.game}NoFrameskip-v4", mode="training", hist_len=4
    )
    print(env)

    # construct an estimator to be used with the policy
    action_no = env.action_space.n
    estimator = get_estimator(
        "atari", hist_len=4, action_no=action_no, hidden_sz=256
    )
    estimator = estimator.cuda()

    # construct an epsilon greedy policy
    # also: epsilon = {'name':'linear', 'start':1, 'end':0.1, 'steps':1000}
    epsilon = get_epsilon(steps=opt.epsilon_steps, end=opt.epsilon_end)
    policy_evaluation = EpsilonGreedyPolicy(estimator, action_no, epsilon)

    # construct a policy improvement type
    # optimizer = get_optimizer('Adam', estimator, lr=0.0001, eps=0.0003)
    optimizer = optim.Adam(
        estimator.parameters(), lr=opt.lr, eps=opt.adam_eps
    )
    policy_improvement = DQNPolicyImprovement(
        estimator, optimizer, gamma=0.99, is_double=opt.double
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
    else:
        experience_replay = ER(1_000_000, batch_size=32)
        # experience_replay = ER(100000, batch_size=32, hist_len=4)  # flat

    # construct a tester
    tester = None

    log = Logger(label="label", path=opt.out_dir)
    train_log = log.add_group(
        tag="training",
        metrics=(
            log.SumMetric("ep_cnt", resetable=False),
            log.AvgMetric("rw_per_ep", emph=True),
            log.AvgMetric("rw_per_step"),
            log.MaxMetric("max_q"),
            log.FPSMetric("training_fps"),
            log.FPSMetric("sampling_fps"),
        ),
        console_options=("white", "on_blue", ["bold"]),
    )
    log.log_info(train_log, "date: %s." % time.strftime("%d/%m/%Y | %H:%M:%S"))
    log.log_info(train_log, "pytorch v%s." % torch.__version__)

    # Add the created objects in the opt namespace
    opt.env = env
    opt.policy_evaluation = policy_evaluation
    opt.policy_improvement = policy_improvement
    opt.experience_replay = experience_replay
    opt.tester = tester
    opt.log = log
    if opt.prioritized:
        opt.priority_update = priority_update_cb

    # print the opt
    print("Starting experiment using the following settings:")
    print(liftoff.config.config_to_string(opt))

    # start the training
    train(opt)


def main():
    """ Read config files and call experiment factories.
    """

    # read config files using liftoff
    opt = read_config()

    # create paths if not using liftoff
    # liftoff takes care of this otherwise
    opt = create_paths(opt)

    # run your experiment
    run(opt)


if __name__ == "__main__":
    main()
