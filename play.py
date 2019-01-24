""" Vizualize agents.
"""
import argparse
import pickle
from pathlib import Path

import torch
import yaml
from termcolor import colored as clr

import liftoff
from liftoff.liftoff_results import collect_all_results

from wintermute.env_wrappers import get_wrapped_atari
from wintermute.estimators import get_estimator
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon


def get_args():
    psr = argparse.ArgumentParser(description="Vizualize agent.")
    psr.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Name of the experiment that that trained an agent you want to see.",
    )
    psr.add_argument(
        "-d",
        "--results_dir",
        default="./results/",
        type=str,
        help="Where are the experiments located.",
    )
    return psr.parse_args()


def get_best_agent(results, verbose=True):
    """ Receives a list of trials and returns the best model it finds across
        trials and time-steps.

        `results` is of the form [(`experiment_path`, [*.yaml, *.pth, ...]), ...]
    """
    if verbose:
        exp_no = len(results)
        models_no = exp_no * 50
        print(f"Searching through {exp_no} experiments, ~{models_no} models.")

    best_score = -float("inf")
    best_agent = None
    for i, (exp_path, file_names) in enumerate(results):
        exp_path = Path(exp_path)
        cfg = yaml.safe_load(open(exp_path / "cfg.yaml", "r"))

        for file in file_names:
            file_path = exp_path / file
            if file_path.suffix == ".pth":
                model = torch.load(file_path)
                rw_per_ep = model["rw_per_ep"]
                print("x", end="", flush=True)
                if rw_per_ep > best_score:
                    best_agent = (cfg, model)
                    best_score = rw_per_ep
                    print("X", end="")

    if verbose:
        cfg, model = best_agent
        msg = (
            f"\n\nBest agent on game {cfg['game']}"
            + f" with rw/ep={model['rw_per_ep']:5.2f} @ step={model['step']}:"
        )
        print(clr(msg, "green"))
        for k, v in cfg.items():
            if k in ("game", "double", "prioritized", "_experiment_parameters"):
                print(clr(k, "yellow"), ": ", v)

        return best_agent


def get_stuff(opt, model):
    # wrap the gym env
    env = get_wrapped_atari(
        opt.game,
        mode="testing",
        seed=42,
        no_gym=opt.no_gym,
        device=opt.mem_device,
    )
    action_no = env.action_space.n
    estimator = get_estimator(
        "atari",
        hist_len=4,
        action_no=action_no,
        hidden_sz=512,
        shared_bias=opt.shared_bias,
    )
    estimator = estimator.cuda()
    estimator.load_state_dict(model["model"])

    epsilon = get_epsilon(name="constant", start=opt.eval_epsilon)
    policy_evaluation = EpsilonGreedyPolicy(estimator, action_no, epsilon)

    return env, policy_evaluation


def play(opt, env, policy_evaluation):
    print(clr("Start playing!", "red"))
    ep_cnt, done = 0, True
    while True:
        if done:
            state, reward, done = env.reset(), 0, False
            ep_rw = 0

        with torch.no_grad():
            pi = policy_evaluation(state)

        state, reward, done, _ = env.step(pi.action)
        env.render()

        ep_rw += reward

        if done:
            print(f"Episode {ep_cnt:3d} done. Total return: {ep_rw:5.2f}.")
            ep_cnt += 1


def main():
    args = get_args()
    results = collect_all_results(
        [args.experiment], results_dir=args.results_dir
    )
    results = [trial for exp in results for trial in exp]

    best_cfg, best_model = get_best_agent(results)
    opt = liftoff.config.dict_to_namespace(best_cfg)
    env, policy_evaluation = get_stuff(opt, best_model)

    play(opt, env, policy_evaluation)


if __name__ == "__main__":
    main()
