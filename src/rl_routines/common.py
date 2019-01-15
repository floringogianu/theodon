""" Functions to be used in all training modes (serial / async).
"""

from typing import Tuple
import pickle
import numpy as np

import torch
from torch import optim

from wintermute.estimators import get_atari_estimator
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from rl_logger import Logger

from src.utils import get_process_memory
from src.bootstrapped import BootstrappedPE
from src.bootstrapped import BootstrappedPI


def init_eval_logger(out_dir=None, log: Logger = None):
    """ Here we initialize the logger used by other functions below.
    """
    if log is None:
        log = Logger(label="label", path=out_dir)
    log.add_group(
        tag="evaluation",
        metrics=(
            log.SumMetric("ep_cnt"),
            log.EpisodicMetric("rw_per_ep", emph=True),
            log.EpisodicMetric("clipped_rw_per_ep", emph=True),
            log.AvgMetric("step_per_ep"),
            log.AvgMetric("rw_per_step"),
            log.MaxMetric("max_q"),
            log.FPSMetric("eval_fps"),
            log.MaxMetric("ram"),
            log.MaxMetric("gpu"),
        ),
        console_options=("white", "on_magenta", ["bold"]),
    )
    return log


def evaluate_once(  # pylint: disable=bad-continuation
    crt_step, policy_evaluation, env, eval_steps, log
) -> Tuple[float, float]:
    """ Here we evaluate a policy.
    """
    eval_log = log.groups["evaluation"]
    eval_log.reset()
    log.log_info(eval_log, f"Start evaluation at {crt_step} training steps.")

    total_rw, total_crw = 0, 0
    nepisodes = 0
    done = True
    crt_return, crt_creturn = 0, 0
    step = 0
    while step < eval_steps or not done:
        if done:
            state, reward, done = env.reset(), 0, False
            crt_return, crt_creturn = 0, 0
        with torch.no_grad():
            pi = policy_evaluation(state)
        state, reward, done, _ = env.step(pi.action)
        creward = min(1, max(-1, reward))

        # do some logging
        eval_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            clipped_rw_per_ep=(creward, (1 if done else 0)),
            step_per_ep=(1, (1 if done else 0)),
            rw_per_step=reward,
            max_q=pi.q_value if pi.q_value else float("-inf"),
            eval_fps=1,
        )
        crt_return += reward
        crt_creturn += creward
        step += 1

        if done:
            nepisodes += 1
            total_rw += crt_return
            total_crw += crt_creturn

    used_ram, used_gpu = get_process_memory()
    eval_log.update(ram=used_ram, gpu=used_gpu)
    log.log_info(eval_log, f"Evaluation results.")
    log.log(eval_log, crt_step)
    eval_log.reset()

    return total_rw / nepisodes, total_crw / nepisodes


def process_eval_results(opt, new_eval_results, best_rw) -> float:
    """Here we process the results of a new evaluation.

       We save the model if needed. The function returns the higher value
       between the previous best reward and the current result.
    """
    eval_step, model, mean_ep_rw, mean_ep_crw = new_eval_results
    eval_log = opt.log.groups["evaluation"]

    if mean_ep_rw > best_rw:
        opt.log.log_info(
            eval_log,
            f"New best model: {mean_ep_rw:8.2f} rw/ep @ {eval_step} steps!",
        )
        torch.save(
            {"rw_per_ep": mean_ep_rw, "step": eval_step, "model": model},
            f"{opt.out_dir}/best_model.pth",
        )
        best_rw = mean_ep_rw
    # save model
    if eval_step % 1_000_000 == 0:
        torch.save(
            {"rw_per_ep": mean_ep_rw, "step": eval_step, "model": model},
            f"{opt.out_dir}/model_{eval_step}.pth",
        )

    if not hasattr(opt, "evals"):
        opt.evals = []
    if not hasattr(opt, "clipped_evals"):
        opt.clipped_evals = []

    opt.evals.append(mean_ep_rw)
    opt.clipped_evals.append(mean_ep_crw)
    summary = {
        "best": best_rw,
        "last-5": np.mean(opt.evals[-5:]),
        "last-10": np.mean(opt.evals[-10:]),
        "best-clip": np.max(opt.clipped_evals),
        "last-5-clip": np.mean(opt.clipped_evals[-5:]),
        "last-10-clip": np.mean(opt.clipped_evals[-10:]),
        "step": eval_step,
    }
    with open(f"{opt.out_dir}/summary.pkl", "wb") as handler:
        pickle.dump(summary, handler, pickle.HIGHEST_PROTOCOL)

    return best_rw


def priority_update(mem, idxs, weights, dqn_loss):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss
    priorities = dqn_loss.priorities.detach().abs()
    mem.update(idxs, [priority.item() for priority in priorities])
    return (losses * weights.to(losses.device).view_as(losses)).mean()


def create_memory(opt):
    """ Initializes experience replay for
    """
    from wintermute.replay import MemoryEfficientExperienceReplay as ER
    from wintermute.replay import PinnedExperienceReplay as PinnedER
    from wintermute.replay.prioritized_replay import ProportionalSampler as PER

    bootstrap_args = None
    if opt.heads_no > 1 and opt.mask and opt.mask < 1:
        bootstrap_args = (opt.heads_no, opt.mask)
    cb = None

    if opt.pinned_memory:
        experience_replay = PinnedER(
            opt.mem_size,
            batch_size=opt.batch_size,
            async_memory=opt.async_memory and not opt.prioritized,
            device=opt.mem_device,
            bootstrap_args=bootstrap_args
        )
    else:
        experience_replay = ER(
            opt.mem_size,
            batch_size=opt.batch_size,
            async_memory=opt.async_memory and not opt.prioritized,
            bootstrap_args=bootstrap_args
        )
    if opt.prioritized:
        experience_replay = PER(
            experience_replay,
            opt.async_memory,
            alpha=0.6,
            beta=0.4,
            optim_steps=((opt.step_no - opt.learn_start) / opt.update_freq),
        )
        cb = priority_update

    return experience_replay, cb


def get_policy_improvement(opt, estimator):
    # construct a policy improvement type
    optimizer = getattr(optim, opt.optimizer_name)(
        estimator.parameters(),
        **opt.optimizer_args.__dict__
    )

    target = create_estimator(opt, estimator.actions_no)
    # construct policy evaluation and policy improvement objects
    if hasattr(opt, "heads_no") and opt.heads_no > 1:
        # for bootstrapping/ensemble methods
        policy_improvement = BootstrappedPI(
            estimator,
            optimizer,
            target,
            **opt.__dict__
        )
    else:
        raise NotImplementedError
        policy_improvement = DQNPolicyImprovement(
            estimator, optimizer, gamma=0.99, is_double=opt.double
        )

    return policy_improvement


def get_policy_evaluation(opt, estimator, action_no, train=True):
    # construct an epsilon greedy policy
    # also: epsilon = {'name':'linear', 'start':1, 'end':0.1, 'steps':1000}

    if train:
        epsilon = get_epsilon(
            steps=opt.epsilon_steps,
            end=opt.epsilon_end,
            warmup_steps=opt.learn_start,
        )
    else:
        epsilon = get_epsilon(name="constant", start=opt.eval_epsilon)

    # construct policy evaluation object
    if hasattr(opt, "heads_no") and opt.heads_no > 1:
        # We need a way to combine several predictions
        policy_evaluation = BootstrappedPE(
            estimator,
            action_no,
            epsilon,
            **opt.__dict__
        )
    else:
        # for all the othe we use classic DQN/DQN objects
        policy_evaluation = EpsilonGreedyPolicy(estimator, action_no, epsilon)

    return policy_evaluation


def create_estimator(opt, action_no):
    estimator = get_atari_estimator(action_no, **opt.atari_estimator.__dict__)
    estimator.to("cuda")
    return estimator
