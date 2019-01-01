""" Entry point.
"""
import time
import random
from copy import deepcopy
from functools import partial
import concurrent.futures

import torch
from torch import optim
from rl_logger import Logger

from wintermute.env_wrappers import get_wrapped_atari
from wintermute.estimators import get_estimator
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon

# from wintermute.policy_improvement import get_optimizer
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import MemoryEfficientExperienceReplay as ER
from wintermute.replay.prioritized_replay import ProportionalSampler as PER

import liftoff
from liftoff.config import read_config

from src.utils import create_paths
from src.utils import get_process_memory


def priority_update(mem, dqn_loss):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss
    mem.update([loss.item() for loss in losses.detach().abs()])
    return (losses * mem.weights.to(losses.device).view_as(losses)).mean()


def async_train(opt):
    """ Here we do the async training.
    """
    env = opt.env
    train_log = opt.log.groups["training"]

    action_space = opt.policy_evaluation.action_space
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    async_update_result = None
    async_test_result = None
    new_test_results = None

    _state, _reward, _done = env.reset(), 0, False
    _action = action_space.sample()
    _next_state, reward, done, _ = env.step(_action)
    state = _next_state

    warmed_up = False
    ep_cnt, best_rw = 0, -float("inf")

    for step in range(1, opt.step_no + 1):

        warmed_up = step > opt.learn_start  # We should learn
        do_training = (step % opt.update_freq == 0) and warmed_up

        if async_update_result is not None:
            async_update_result.result()  # Wait for policy improvement
            async_update_result = None

        if step % opt.target_update == 0 and warmed_up:
            opt.policy_improvement.update_target_estimator()

        with torch.no_grad():
            pi = opt.policy_evaluation(state)

        # The code below is actually async

        if do_training:
            batch = opt.experience_replay.push_and_sample(
                (_state, _action, _reward, _next_state, _done)
            )
            if opt.prioritized:
                async_update_result = executor.submit(
                    opt.policy_improvement, batch, cb=opt.priority_update
                )
            else:
                async_update_result = executor.submit(opt.policy_improvement, batch)

        else:
            opt.experience_replay.push((_state, _action, _reward, _next_state, _done))

        next_state, reward, done, _ = env.step(pi.action)
        _state, _action, _reward, _next_state, _done = (
            state,
            pi.action,
            reward,
            next_state,
            done,
        )
        state = next_state

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
                used_ram, used_gpu = get_process_memory()
                train_log.update(ram=used_ram, gpu=used_gpu)

                opt.log.log(train_log, step)
                train_log.reset()

        # just testing below

        if async_test_result is not None:
            # pylint: disable=E0633
            test_step, test_estimator, result = async_test_result
            if result.done():
                mean_ep_rw = result.result()
                new_test_results = test_step, test_estimator, mean_ep_rw
                async_test_result = None

        if step % opt.test_freq == 0:
            if async_update_result is not None:
                async_update_result.result()
                async_update_result = None

            if opt.async_eval:
                if async_test_result is not None:
                    # Wait for the previous evaluation to end
                    test_step, test_estimator, result = async_test_result
                    mean_ep_rw = result.result()
                    new_test_results = test_step, test_estimator, mean_ep_rw

                _estimator = deepcopy(opt.policy_evaluation.policy.estimator)
                result = executor.submit(
                    test, opt, step, _estimator, action_space, opt.test_env, opt.log
                )
                async_test_result = (step, _estimator, result)
            else:
                test_estimator = deepcopy(opt.policy_evaluation.policy.estimator)
                test_step = step
                mean_ep_rw = test(
                    opt, step, test_estimator, action_space, opt.test_env, opt.log
                )
                new_test_results = test_step, test_estimator, mean_ep_rw

        if new_test_results is not None:
            best_rw = process_test_results(opt, new_test_results, best_rw)
            new_test_results = None

    if async_test_result is not None:
        # pylint: disable=E0633
        test_step, test_estimator, result = async_test_result
        mean_ep_rw = result.result()
        new_test_results = test_step, test_estimator, mean_ep_rw
        best_rw = process_test_results(opt, new_test_results, best_rw)

    opt.log.log(train_log, step)
    train_log.reset()


def train(opt):
    """ Here we do the training.
    """
    env = opt.env
    train_log = opt.log.groups["training"]

    async_test_result = None  # type: Optional[tuple]
    new_test_results = None  # type: Tuple[int, nn.Module, float]

    action_space = opt.policy_evaluation.action_space
    if opt.async_eval:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    state, reward, done = env.reset(), 0, False
    warmed_up = False
    ep_cnt, best_rw = 0, -float("inf")
    for step in range(1, opt.step_no + 1):

        # take action and save the s to _s and a to _a to be used later
        with torch.no_grad():
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

        if step % opt.target_update == 0 and warmed_up:
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
                used_ram, used_gpu = get_process_memory()
                train_log.update(ram=used_ram, gpu=used_gpu)
                opt.log.log(train_log, step)
                train_log.reset()

        warmed_up = step > opt.learn_start

        # testing

        if async_test_result is not None:
            # pylint: disable=E0633
            test_step, test_estimator, result = async_test_result
            if result.done():
                mean_ep_rw = result.result()
                new_test_results = test_step, test_estimator, mean_ep_rw
                async_test_result = None

        if step % opt.test_freq == 0:
            if opt.async_eval:
                if async_test_result is not None:
                    # Wait for the previous evaluation to end
                    test_step, test_estimator, result = async_test_result
                    mean_ep_rw = result.result()
                    new_test_results = test_step, test_estimator, mean_ep_rw

                _estimator = deepcopy(opt.policy_evaluation.policy.estimator)
                result = executor.submit(
                    test, opt, step, _estimator, action_space, opt.test_env, opt.log
                )
                async_test_result = (step, _estimator, result)
            else:
                test_estimator = deepcopy(opt.policy_evaluation.policy.estimator)
                test_step = step
                mean_ep_rw = test(
                    opt, step, test_estimator, action_space, opt.test_env, opt.log
                )
                new_test_results = test_step, test_estimator, mean_ep_rw

        if new_test_results is not None:
            best_rw = process_test_results(opt, new_test_results, best_rw)
            new_test_results = None

    if async_test_result is not None:
        # pylint: disable=E0633
        test_step, test_estimator, result = async_test_result
        mean_ep_rw = result.result()
        new_test_results = test_step, test_estimator, mean_ep_rw
        best_rw = process_test_results(opt, new_test_results, best_rw)

    opt.log.log(train_log, step)
    train_log.reset()


def process_test_results(opt, new_test_results, best_rw) -> float:
    """Here we process the results of a new evaluation.

       We save the model if needed. The function returns the higher value
       between the previous best reward and the current result.
    """
    test_step, test_estimator, mean_ep_rw = new_test_results
    train_log = opt.log.groups["training"]
    # save best model
    model = test_estimator.state_dict()
    if mean_ep_rw > best_rw:
        opt.log.log_info(
            train_log, f"New best model: {mean_ep_rw:8.2f} rw/ep @ {test_step} steps!"
        )
        torch.save(
            {"rw_per_ep": mean_ep_rw, "step": test_step, "model": model},
            f"{opt.out_dir}/best_model.pth",
        )
        best_rw = mean_ep_rw
    # save model
    if test_step % 1_000_000 == 0:
        torch.save(
            {"rw_per_ep": mean_ep_rw, "step": test_step, "model": model},
            f"{opt.out_dir}/model_{test_step}.pth",
        )
    return best_rw


def test(opt, crt_step, estimator, action_space, env, log):
    """ Here we do the training.

        DeepMind uses a constant epsilon schedule with a very small value
        instead  of a completely Deterministic Policy.
    """

    epsilon = get_epsilon(name="constant", start=opt.test_epsilon)
    policy_evaluation = EpsilonGreedyPolicy(estimator, action_space, epsilon)

    test_log = log.groups["testing"]
    log.log_info(test_log, f"Start testing at {crt_step} training steps.")

    total_rw = 0
    nepisodes = 0
    done = True
    crt_return = 0
    for _ in range(1, opt.test_steps + 1):
        if done:
            nepisodes += 1
            total_rw += crt_return
            state, reward, done = env.reset(), 0, False
            crt_return = 0
        with torch.no_grad():
            pi = policy_evaluation(state)
        state, reward, done, _ = env.step(pi.action)

        # do some logging
        test_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            step_per_ep=(1, (1 if done else 0)),
            rw_per_step=reward,
            max_q=pi.q_value,
            test_fps=1,
        )
        crt_return += reward

    log.log_info(test_log, f"Evaluation results.")
    log.log(test_log, crt_step)
    test_log.reset()

    return (total_rw / nepisodes) if nepisodes > 0 else crt_return


def run(opt):
    """ Here we initialize stuff.
    """
    opt.seed = random.randint(0, 1e4) if not opt.seed else opt.seed
    print(f"torch manual seed={opt.seed}.")
    torch.manual_seed(opt.seed)

    # wrap the gym env
    env = get_wrapped_atari(opt.game, mode="training", seed=opt.seed, no_gym=opt.no_gym, device=torch.device("cuda"))
    test_env = get_wrapped_atari(
        opt.game, mode="testing", seed=opt.seed, no_gym=opt.no_gym
    )

    # construct an estimator to be used with the policy
    action_no = env.action_space.n
    estimator = get_estimator(
        "atari",
        hist_len=4,
        action_no=action_no,
        hidden_sz=512,
        init_method=opt.init_method,
    )
    estimator = estimator.cuda()

    # construct an epsilon greedy policy
    # also: epsilon = {'name':'linear', 'start':1, 'end':0.1, 'steps':1000}
    epsilon = get_epsilon(steps=opt.epsilon_steps, end=opt.epsilon_end)
    policy_evaluation = EpsilonGreedyPolicy(estimator, action_no, epsilon)

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
        experience_replay = ER(
            opt.mem_size, batch_size=32, async_memory=opt.async_memory
        )
        # experience_replay = ER(100000, batch_size=32, hist_len=4)  # flat

    log = Logger(label="label", path=opt.out_dir)
    train_log = log.add_group(
        tag="training",
        metrics=(
            log.SumMetric("ep_cnt"),
            log.AvgMetric("rw_per_ep", emph=True),
            log.AvgMetric("rw_per_step"),
            log.MaxMetric("max_q"),
            log.FPSMetric("training_fps"),
            log.FPSMetric("sampling_fps"),
            log.MaxMetric("ram"),
            log.MaxMetric("gpu"),
        ),
        console_options=("white", "on_blue", ["bold"]),
    )
    _test_log = log.add_group(
        tag="testing",
        metrics=(
            log.SumMetric("ep_cnt"),
            log.AvgMetric("rw_per_ep", emph=True),
            log.AvgMetric("step_per_ep"),
            log.AvgMetric("rw_per_step"),
            log.MaxMetric("max_q"),
            log.FPSMetric("test_fps"),
        ),
        console_options=("white", "on_magenta", ["bold"]),
    )
    log.log_info(train_log, "date: %s." % time.strftime("%d/%m/%Y | %H:%M:%S"))
    log.log_info(train_log, "pytorch v%s." % torch.__version__)

    # Add the created objects in the opt namespace
    opt.env = env
    opt.test_env = test_env
    opt.policy_evaluation = policy_evaluation
    opt.policy_improvement = policy_improvement
    opt.experience_replay = experience_replay
    opt.log = log
    if opt.prioritized:
        opt.priority_update = priority_update_cb

    # print the opt
    print("Starting experiment using the following settings:")
    print(liftoff.config.config_to_string(opt))

    # start the training
    if opt.async_update:
        async_train(opt)
    else:
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
