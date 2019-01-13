""" Multiprocessing entry point.
"""

# pylint: disable=wrong-import-order

from src.rl_routines import init_evaluator, init_learner, init_player

if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    from copy import deepcopy
    import random
    from torch import multiprocessing as mp

    from wintermute.env_wrappers import get_wrapped_atari
    from src.rl_routines.common import create_estimator

    def launch_processes(opt):
        """ This is the main process that starts the rest.
        """

        opt.seed = random.randint(0, 1e4) if not opt.seed else opt.seed
        print(f"torch manual seed={opt.seed}.")
        torch.manual_seed(opt.seed)

        play_opt = deepcopy(opt)
        learn_opt = deepcopy(opt)
        eval_opt = deepcopy(opt)

        action_no = get_wrapped_atari(
            opt.game, no_gym=opt.no_gym, seed=opt.seed
        ).action_space.n
        estimator = create_estimator(opt, action_no)
        estimator.share_memory()

        play_opt.estimator = estimator
        learn_opt.estimator = estimator

        # Prepare general stuff

        experience_queue = mp.Queue()
        sync_queue = mp.Queue()
        eval_queue = mp.Queue()
        confirm_queue = mp.Queue()

        play_proc = mp.Process(
            target=init_player, args=(play_opt, experience_queue, sync_queue)
        )
        play_proc.start()

        learn_proc = mp.Process(
            target=init_learner,
            args=(
                learn_opt,
                experience_queue,
                sync_queue,
                eval_queue,
                confirm_queue,
            ),
        )
        learn_proc.start()

        eval_proc = mp.Process(
            target=init_evaluator, args=(eval_opt, eval_queue, confirm_queue)
        )
        eval_proc.start()

        learn_proc.join()
        eval_proc.join()
        play_proc.join()

    def main():
        """ Reads configuration.
        """
        from liftoff.config import read_config
        from src.utils import create_paths

        opt = read_config()
        opt = create_paths(opt)

        launch_processes(opt)

    main()
