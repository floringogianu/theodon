""" Multiprocessing entry point.
"""

# pylint: disable=wrong-import-order

from rl_routines import init_evaluator, init_learner, init_player

if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    from argparse import Namespace
    import random
    from torch import multiprocessing as mp

    from wintermute.env_wrappers import get_wrapped_atari
    from wintermute.estimators import get_estimator

    def launch_processes(opt):
        """ This is the main process that starts the rest.
        """

        opt.seed = random.randint(0, 1e4) if not opt.seed else opt.seed
        print(f"torch manual seed={opt.seed}.")
        torch.manual_seed(opt.seed)

        play_opt = Namespace(
            out_dir=opt.out_dir,
            game=opt.game,
            seed=opt.seed,
            step_no=opt.step_no,
            no_gym=opt.no_gym,
            epsilon_steps=opt.epsilon_steps,
            epsilon_end=opt.epsilon_end,
            update_freq=opt.update_freq,
            target_update=opt.target_update,
            learn_start=opt.learn_start,
            eval_freq=opt.eval_freq,
            log_freq=opt.log_freq,
        )
        learn_opt = Namespace(
            out_dir=opt.out_dir,
            hist_len=opt.hist_len,
            lr=opt.lr,
            double=opt.double,
            prioritized=opt.prioritized,
            mem_size=opt.mem_size,
            pinned_memory=opt.pinned_memory,
            async_memory=opt.async_memory,
            learn_log_freq=opt.learn_log_freq,
            learn_start=opt.learn_start
        )
        if opt.prioritized:
            learn_opt.priority_update_cb = opt.priority_update_cb

        eval_opt = Namespace(
            out_dir=opt.out_dir,
            game=opt.game,
            seed=opt.seed,
            no_gym=opt.no_gym,
            hidden_sz=opt.hidden_sz,
            hist_len=opt.hist_len,
            eval_epsilon=opt.eval_epsilon,
            eval_steps=opt.eval_steps,
        )

        action_no = get_wrapped_atari(
            opt.game, no_gym=opt.no_gym, seed=opt.seed
        ).action_space.n
        estimator = get_estimator(
            "atari",
            hist_len=opt.hist_len,
            action_no=action_no,
            hidden_sz=opt.hidden_sz,
        )
        estimator = estimator.cuda()
        estimator.share_memory()

        play_opt.estimator = estimator
        learn_opt.estimator = estimator
        # eval_opt.estimator = estimator

        # Prepare general stuff

        experience_queue = mp.Queue()
        sync_queue = mp.Queue()
        eval_queue = mp.Queue()

        play_proc = mp.Process(
            target=init_player,
            args=(play_opt, experience_queue, sync_queue, eval_queue),
        )
        play_proc.start()

        learn_proc = mp.Process(
            target=init_learner, args=(learn_opt, experience_queue, sync_queue)
        )
        learn_proc.start()

        eval_proc = mp.Process(
            target=init_evaluator, args=(eval_opt, eval_queue)
        )
        eval_proc.start()

        play_proc.join()
        learn_proc.join()
        eval_proc.join()

    def main():
        """ Reads configuration.
        """
        from liftoff.config import read_config
        from src.utils import create_paths

        opt = read_config()
        opt = create_paths(opt)

        launch_processes(opt)

    main()
