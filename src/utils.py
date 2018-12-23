""" Utility functions.
"""

import os
import argparse
from datetime import datetime


def create_paths(args: argparse.Namespace) -> argparse.Namespace:
    """ Creates directories containing the results of the experiments when you
        use `python main.py` instead of `liftoff main.py`. It uses liftoff
        convention for the folder names: `timestamp_experimentname`.
    """
    time_stamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    print(time_stamp)
    if not hasattr(args, "out_dir") or args.out_dir is None:
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        out_dir = f"./results/{time_stamp}_{args.experiment:s}"
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    return args


def get_parser(**kwargs) -> argparse.Namespace:
    """ Configure a parser with custom defaults.

    Returns:
        Namespace: Contains the parsed arguments.
    """

    prs = argparse.ArgumentParser(description=kwargs.get("description", "DQN"))
    prs.add_argument("--seed", type=int, default=42, help="RNG seed.")
    prs.add_argument(
        "--game",
        type=str,
        default=kwargs.get("game", "SpaceInvaders"),
        help="ATARI game.",
    )
    prs.add_argument(
        "--label",
        type=str,
        default="",
        help="Experiment label, used in the naming of folders.",
    )
    prs.add_argument(
        "--step_no",
        type=int,
        default=kwargs.get("step_no", 50_000_000),
        help="Total no of training steps.",
    )
    prs.add_argument(
        "--double-dqn",
        action=kwargs.get("double_dqn", "store_true"),
        help="Train with Double-DQN.",
    )
    prs.add_argument(
        "--prioritized",
        action=kwargs.get("prioritized", "store_true"),
        help="Train with Prioritized Experience Replay.",
    )
    prs.add_argument(
        "--lr",
        type=float,
        default=kwargs.get("lr", 0.00025),
        help="Adam learning rate.",
        metavar="η",
    )
    prs.add_argument(
        "--adam-eps",
        type=float,
        default=1.5e-4,
        metavar="ε",
        help="Adam epsilon",
    )
    prs.add_argument(
        "--update-freq",
        type=int,
        default=kwargs.get("update_freq", 4),
        help="How often we optimized the model.",
    )
    prs.add_argument(
        "--epsilon-steps",
        type=int,
        default=kwargs.get("epsilon_steps", 1_000_000),
        help="No of exploration steps.",
    )
    prs.add_argument(
        "--learn-start",
        type=int,
        default=kwargs.get("learn_start", 80000),
        help="No of steps after which the learning starts.",
    )
    prs.add_argument(
        "--mem-size",
        type=int,
        default=kwargs.get("mem_size", 1_000_000),
        help="Size of the Experience Replay.",
    )
    prs.add_argument(
        "--log-freq",
        type=int,
        default=kwargs.get("log_freq", 10),
        help="How often do we log to the console and save the results.",
    )
    return prs.parse_args()
