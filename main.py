""" Entry point.
"""
# pylint: disable=wrong-import-order
import torch

if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    import time
    import random
    from copy import deepcopy
    from functools import partial
    from argparse import Namespace
    import concurrent.futures

    import liftoff
    from liftoff.config import read_config

    from src.utils import create_paths  # TODO: change
    from src.utils import get_process_memory

from wintermute.env_wrappers import get_wrapped_atari
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from src.bootstrapped import BootstrappedPE
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon

from src.rl_routines.common import (
    init_eval_logger,
    process_eval_results,
    evaluate_once,
    create_memory,
    get_policy_evaluation,
    get_policy_improvement,
    create_estimator
)

if __name__ == "__main__":

    def train(opt):
        """ Here we do the training.
        """
        env = opt.env
        train_log = opt.log.groups["training"]
        train_log.reset()

        async_eval_result = None  # type: Optional[tuple]
        new_eval_results = None  # type: Tuple[int, nn.Module, float]

        action_space = env.action_space.n
        if opt.async_eval:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

        thompson_sampling = False
        if hasattr(opt, "boot") and opt.boot.k > 1:
            thompson_sampling = opt.boot.is_thompson
        if thompson_sampling:
            ts_idx = opt.policy_evaluation.policy.sample_posterior_idx()
            opt.policy_improvement.set_posterior_idx(ts_idx)

        state, reward, done = env.reset(), 0, False
        warmed_up = False
        ep_cnt, best_rw = 0, -float("inf")
        eval_estimator = None
        for step in range(1, opt.step_no + 1):

            # take action and save the s to _s and a to _a to be used later
            with torch.no_grad():
                pi = opt.policy_evaluation(state)
            _state, _action = state, pi.action
            state, reward, done, _ = env.step(pi.action)

            # add a (_s, _a, r, d) transition
            opt.experience_replay.push((_state, pi, reward, state, done))

            # sample a batch and do some learning
            do_training = (step % opt.update_freq == 0) and warmed_up

            if do_training:
                batch = opt.experience_replay.sample()
                if opt.prioritized:  # batch is (data, idxs, weights)
                    data, idxs, weights = batch
                    update_priorities = partial(
                        opt.cb, opt.experience_replay, idxs, weights
                    )
                    opt.policy_improvement(data, cb=update_priorities)
                else:
                    opt.policy_improvement(batch)

            if step % opt.target_update == 0 and warmed_up:
                opt.policy_improvement.update_target_estimator()

            # do some logging
            train_log.update(
                ep_cnt=(1 if done else 0),
                rw_per_ep=(reward, (1 if done else 0)),
                rw_per_step=reward,
                max_q=pi.q_value if pi.q_value else float("-inf"),
                sampling_fps=1,
                training_fps=32 if do_training else 0,
            )

            if done:
                state, reward, done = env.reset(), 0, False
                ep_cnt += 1
                if thompson_sampling:
                    ts_idx = opt.policy_evaluation.policy.sample_posterior_idx()
                    opt.policy_improvement.set_posterior_idx(ts_idx)

                if ep_cnt % opt.log_freq == 0:
                    used_ram, used_gpu = get_process_memory()
                    train_log.update(ram=used_ram, gpu=used_gpu)
                    opt.log.log(train_log, step)
                    train_log.reset()

            warmed_up = step > opt.learn_start

            # testing

            if async_eval_result is not None:
                # pylint: disable=E0633
                eval_step, _eval_estimator, result = async_eval_result
                if result.done():
                    avgs = result.result()
                    new_eval_results = (
                        eval_step,
                        _eval_estimator.state_dict(),
                        *avgs,
                    )
                    async_eval_result = None

            if step % opt.eval_freq == 0:
                if opt.async_eval:
                    if async_eval_result is not None:
                        # Wait for the previous evaluation to end
                        eval_step, _eval_estimator, result = async_eval_result
                        avgs = result.result()
                        new_eval_results = (
                            eval_step,
                            _eval_estimator.state_dict(),
                            *avgs,
                        )

                    eval_estimator = deepcopy(
                        opt.policy_evaluation.policy.estimator
                    ).cpu()
                    result = executor.submit(
                        test,
                        opt.eval_opt,
                        step,
                        eval_estimator,
                        action_space,
                        None,  # do not pickle eval_env if evaluation is async
                        opt.log,
                    )
                    async_eval_result = (step, eval_estimator, result)
                else:
                    eval_estimator = deepcopy(
                        opt.policy_evaluation.policy.estimator
                    )
                    eval_step = step
                    avgs = test(
                        opt.eval_opt,
                        step,
                        eval_estimator,
                        action_space,
                        opt.eval_env,
                        opt.log,
                    )
                    new_eval_results = (
                        eval_step,
                        eval_estimator.state_dict(),
                        *avgs,
                    )

            if new_eval_results is not None:
                best_rw = process_eval_results(opt, new_eval_results, best_rw)
                new_eval_results = None

        if async_eval_result is not None:
            # pylint: disable=E0633
            eval_step, eval_estimator, result = async_eval_result
            avgs = result.result()
            new_eval_results = (eval_step, eval_estimator.state_dict(), *avgs)
            best_rw = process_eval_results(opt, new_eval_results, best_rw)

        try:
            opt.log.log(train_log, step)
        except ZeroDivisionError:
            pass  # It's ok, nothing really new to report here
        train_log.reset()


if __name__ in ["__mp_main__", "__main__"]:

    def test(opt, crt_step, estimator, action_space, eval_env, log):
        """ Here we do the training.

            DeepMind uses a constant epsilon schedule with a very small value
            instead  of a completely Deterministic Policy.
        """

        epsilon = get_epsilon(name="constant", start=opt.eval_epsilon)
        estimator.to("cuda")

        if hasattr(opt, "boot") and opt.boot.k > 1:
            policy_evaluation = BootstrappedPE(
                estimator,
                action_space,
                epsilon,
                is_thompson=False,
                vote=opt.boot.vote,
                test=True,
            )
        else:
            policy_evaluation = EpsilonGreedyPolicy(
                estimator, action_space, epsilon
            )

        if eval_env is None:
            eval_env = get_wrapped_atari(
                opt.game, mode="testing", seed=opt.seed, no_gym=opt.no_gym
            )

        mean_ep_rw, mean_ep_crw = evaluate_once(
            crt_step, policy_evaluation, eval_env, opt.eval_steps, log
        )

        return mean_ep_rw, mean_ep_crw


if __name__ == "__main__":

    def run(opt):
        """ Here we initialize stuff.
        """
        opt.seed = random.randint(0, 1e4) if not opt.seed else opt.seed
        print(f"torch manual seed={opt.seed}.")
        torch.manual_seed(opt.seed)

        # wrap the gym env
        env = get_wrapped_atari(
            opt.game,
            mode="training",
            seed=opt.seed,
            no_gym=opt.no_gym,
            device=opt.mem_device,
        )

        if opt.async_eval:
            eval_env = None
        else:
            eval_env = get_wrapped_atari(
                opt.game, mode="testing", seed=opt.seed, no_gym=opt.no_gym
            )

        # construct an estimator to be used with the policy
        action_no = env.action_space.n
        estimator = create_estimator(opt, action_no)

        # construct DQN/BootstrappedDQN Policy Evaluation and Improvement
        policy_evaluation = get_policy_evaluation(opt, estimator, action_no)
        policy_improvement = get_policy_improvement(opt, estimator)

        # Get Experience Replay and a callback function for updating priorities
        experience_replay, update_priorities = create_memory(opt)

        log = init_eval_logger(opt.out_dir)
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

        log.log_info(
            train_log, "date: %s." % time.strftime("%d/%m/%Y | %H:%M:%S")
        )

        log.log_info(train_log, "pytorch v%s." % torch.__version__)

        # Add the created objects in the opt namespace
        opt.env = env
        opt.eval_env = eval_env
        opt.policy_evaluation = policy_evaluation
        opt.policy_improvement = policy_improvement
        opt.experience_replay = experience_replay
        opt.cb = update_priorities
        opt.log = log

        # print the opt
        print("Starting experiment using the following settings:")
        print(liftoff.config.config_to_string(opt))
        print(estimator)

        opt.eval_opt = Namespace(
            eval_steps=opt.eval_steps,
            eval_epsilon=opt.eval_epsilon,
            game=opt.game,
            seed=opt.seed,
            no_gym=opt.no_gym,
            boot=opt.boot if hasattr(opt, "boot") else None,
        )

        opt.evals = []

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
