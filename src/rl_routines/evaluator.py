""" Here are the functions for the async evaluator.
"""

from wintermute.env_wrappers import get_wrapped_atari
from wintermute.estimators import get_estimator
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon

from .common import evaluate_once, process_eval_results, init_eval_logger

def evaluate(opt):
    """ Here we wait for some other process to send us a state dict to eval it.
    """

    eval_queue = opt.eval_queue
    confirm_queue = opt.confirm_queue
    env = opt.env
    policy_evaluation = opt.policy_evaluation

    best_rw = float("-inf")
    msg = eval_queue.get()
    while isinstance(msg, tuple):
        step, state_dict = msg
        policy_evaluation.policy.estimator.load_state_dict(state_dict)
        policy_evaluation.policy.estimator.cuda()
        mean_ep_rw, mean_ep_crw = evaluate_once(
            step, policy_evaluation, env, opt.eval_steps, opt.log
        )
        best_rw = process_eval_results(
            opt, (step, state_dict, mean_ep_rw, mean_ep_crw), best_rw
        )
        confirm_queue.put(best_rw)
        msg = eval_queue.get()
    return best_rw


def init_evaluator(opt, eval_queue, confirm_queue):
    """ Here we initialize the evaluator, creating objects and shit.
    """

    log = init_eval_logger(opt.out_dir)

    env = get_wrapped_atari(
        opt.game, mode="testing", seed=opt.seed, no_gym=opt.no_gym
    )

    eval_estimator = get_estimator(
        "atari",
        hist_len=opt.hist_len,
        action_no=env.action_space.n,
        hidden_sz=opt.hidden_sz,
        shared_bias=opt.shared_bias,
    )
    eval_estimator.cuda()

    epsilon = get_epsilon(name="constant", start=opt.eval_epsilon)
    policy_evaluation = EpsilonGreedyPolicy(
        eval_estimator, env.action_space, epsilon
    )

    opt.log = log
    opt.env = env
    opt.policy_evaluation = policy_evaluation
    opt.eval_queue = eval_queue
    opt.confirm_queue = confirm_queue

    evaluate(opt)
