""" Functions to be used in all training modes (serial / async).
"""

import pickle
import numpy as np
import torch
from src.utils import get_process_memory


def evaluate_once(crt_step, policy_evaluation, env, eval_steps, log) -> float:
    """ Here we evaluate a policy.
    """
    eval_log = log.groups["evaluation"]
    eval_log.reset()
    log.log_info(eval_log, f"Start evaluation at {crt_step} training steps.")

    total_rw = 0
    nepisodes = 0
    done = True
    crt_return = 0
    step = 0
    while step < eval_steps or not done:
        if done:
            state, reward, done = env.reset(), 0, False
            crt_return = 0
        with torch.no_grad():
            pi = policy_evaluation(state)
        state, reward, done, _ = env.step(pi.action)

        # do some logging
        eval_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            step_per_ep=(1, (1 if done else 0)),
            rw_per_step=reward,
            max_q=pi.q_value,
            eval_fps=1,
        )
        crt_return += reward
        step += 1

        if done:
            nepisodes += 1
            total_rw += crt_return

    used_ram, used_gpu = get_process_memory()
    eval_log.update(ram=used_ram, gpu=used_gpu)
    log.log_info(eval_log, f"Evaluation results.")
    log.log(eval_log, crt_step)
    eval_log.reset()

    return total_rw / nepisodes


def process_eval_results(opt, new_eval_results, best_rw) -> float:
    """Here we process the results of a new evaluation.

       We save the model if needed. The function returns the higher value
       between the previous best reward and the current result.
    """
    eval_step, model, mean_ep_rw = new_eval_results
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
    opt.evals.append(mean_ep_rw)
    summary = {
        "best": best_rw,
        "last-5": np.mean(opt.evals[-5:]),
        "last-10": np.mean(opt.evals[-10:]),
        "step": eval_step,
    }
    with open(f"{opt.out_dir}/summary.pkl", "wb") as handler:
        pickle.dump(summary, handler, pickle.HIGHEST_PROTOCOL)

    return best_rw


def priority_update(mem, dqn_loss):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss
    mem.update([loss.item() for loss in losses.detach().abs()])
    return (losses * mem.weights.to(losses.device).view_as(losses)).mean()
