""" Policy Evaluation and Policy Improvement objects for Bootstrapped
    estimators.
"""
from functools import partial
from typing import NamedTuple

import torch
from gym.spaces import Discrete

from wintermute.policy_improvement import DQNPolicyImprovement, get_dqn_loss
from wintermute.policy_evaluation.exploration_schedules import get_schedule


class DeterministicEnsembleOutput(NamedTuple):
    """ The output of the deterministic policy. """

    action: int
    q_value: float
    sigma: object
    full: object
    posterior: object
    vote_cnt: object


class BootstrappedPE:
    """ Implements the policy evaluation step for bootstrapped estimators
        (ensembles). It has two behaviours:

        1. Approximates Thompson Sampling by picking one ensemble component
        at the begining of an episode and evaluating the policy. For this it
        provides the `sample_posterior_idx` method.

        2. The entire ensemble is used for taking actions. This behaviour
        happens on `is_thompson=False` and it has two modes:
            i) The action is the argmax of the Q-values obtained by averaging
            over the ensemble.
            ii) The action is the one picked by the majority of the ensemble
            components (`vote`==True).
    """

    def __init__(self, estimator, is_thompson=False, vote=True):
        self.__estimator = estimator
        self.__posterior_idx = None
        self.__vote = vote
        self.__ensemble_sz = len(self.__estimator)

        # TODO: check this for batch inputs also
        if is_thompson:
            self.get_action = self.__thompson_action
        elif vote:
            self.get_action = self.__voting_action
        else:
            self.get_action = self.__mean_action

    def __thompson_action(self, state):
        assert (
            self.__posterior_idx is not None
        ), "Call `sample_posterior_idx` first."

        ensemble_qvals = self.__estimator(state).squeeze()
        qvals = ensemble_qvals[self.__posterior_idx]
        qval, argmax_a = qvals.max(0)

        return DeterministicEnsembleOutput(
            action=argmax_a.item(),
            sigma=ensemble_qvals.var(0)[argmax_a].item(),
            q_value=qval.item(),
            full=ensemble_qvals,
            posterior=self.__posterior_idx,
            vote_cnt=False,
        )

    def __mean_action(self, state):
        ensemble_qvals = self.__estimator(state).squeeze()
        qvals = ensemble_qvals.mean(0)
        qval, argmax_a = qvals.max(0)

        return DeterministicEnsembleOutput(
            action=argmax_a.item(),
            sigma=ensemble_qvals.var(0)[argmax_a].item(),
            q_value=qval.item(),
            full=ensemble_qvals,
            posterior="mean",
            vote_cnt=False,
        )

    def __voting_action(self, state):
        ensemble_qvals = self.__estimator(state).squeeze()
        act_no = ensemble_qvals.shape[1]

        qvals, argmaxes = ensemble_qvals.max(1)
        votes = [(argmaxes == act).sum(0) for act in range(act_no)]
        vote_cnt, argmax_a = torch.stack(votes, 0).max(0)

        return DeterministicEnsembleOutput(
            action=argmax_a.item(),
            sigma=ensemble_qvals.var(0)[argmax_a].item(),
            q_value=qvals[argmaxes == argmax_a].mean(),
            full=ensemble_qvals,
            posterior="vote",
            vote_cnt=vote_cnt,
        )

    def sample_posterior_idx(self):
        """ Samples one of the components of the ensemble and returns its
            index.
        """
        self.__posterior_idx = torch.randint(0, self.__ensemble_sz, (1,)).item()
        return self.__posterior_idx


class BootstrappedPI(DQNPolicyImprovement):
    """ Manages the policy improvement step for bootstrapped estimators
        (ensembles). It has three behaviours:

        1. Trains one ensemble component per episode, as in the original
        paper. For this mode set `is_thompson=True` in the constructor and
        call `posterior_idx(idx)` at the begining of each episode during
        training. (Deep Exploration via Bootstrapped
        DQN)[https://arxiv.org/abs/1602.04621]

        2. Trains all the heads all the time. Bootstrapping is achieved only
        through masking transitions and training each ensemble component on its
        assigned data only. Set `is_thompson=False`.

        3. Trains all the heads all the time without masking.

    Args:
        DQNPolicyImprovement (DQNPolicyImprovement): The default DQN update.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        estimator,
        optimizer,
        gamma,
        target_estimator=None,
        is_double=False,
        is_thompson=False,
    ):
        super().__init__(
            estimator, optimizer, gamma, target_estimator, is_double
        )
        self.__is_thompson = is_thompson
        self.__posterior_idx = None

    def __call__(self, batch, cb=None):
        """ Overwrites the parent's methods.
        """
        batch_sz = batch[0].shape[0]
        batch = [el.to(self.device) for el in batch]

        boot_masks = None
        if len(batch) == 2:
            # usual (s,a,r,s_,d) batch, mask of size K * batch_size
            batch, boot_masks = batch

        # scenario 1: sampled component, all data
        if self.__is_thompson:
            idx = self.__posterior_idx
            dqn_loss = get_dqn_loss(
                batch,
                partial(self.estimator, mid=idx),
                self.gamma,
                target_estimator=partial(self.target_estimator, mid=idx),
                is_double=self.is_double,
            ).loss

        # scenario 2: all ensemble components, masked data
        elif boot_masks is not None:
            # split batch in mini-batches for each ensemble component.
            batches = [[el[bm] for el in batch] for bm in boot_masks]

            # Gather the losses for each batch and ensemble component. We use
            # partial application to set which ensemble component gets trained.
            dqn_losses = [
                get_dqn_loss(
                    batch_,
                    partial(self.estimator, mid=mid),
                    self.gamma,
                    target_estimator=partial(self.target_estimator, mid=mid),
                    is_double=self.is_double,
                ).loss
                for mid, batch_ in enumerate(batches)
            ]

            # recompose the dqn_loss
            dqn_loss = torch.zeros((batch_sz, 1), device=dqn_losses[0].device)
            for boot_mask, loss in zip(dqn_losses, boot_masks):
                dqn_loss[boot_mask] += loss

            # TODO: gradient rescalling

        # scenario 3: all ensemble components, all data
        elif boot_masks is None:
            dqn_losses = [
                get_dqn_loss(
                    batch,
                    partial(self.estimator, mid=idx),
                    self.gamma,
                    target_estimator=partial(self.target_estimator, mid=idx),
                    is_double=self.is_double,
                ).loss
                for idx in range(len(self.estimator))
            ]

            # Mean of the losses of each ensemble component on the full batch
            dqn_loss = torch.cat(dqn_losses, dim=1).mean(dim=1).unsqueeze(1)

        if cb:
            loss = cb(dqn_loss)
        else:
            loss = dqn_loss.mean()

        loss.backward()
        self.update_estimator()

    def set_posterior_idx(self, idx):
        assert self.__is_thompson, (
            "Calling this setter means you want to train the ensemble in a "
            + "Thompson sampling setup but you didn't set `is_thompson` in the "
            + "constructor."
        )
        self.__posterior_idx = idx


if __name__ == "__main__":
    from wintermute.estimators import AtariNet, BootstrappedAtariNet

    net = AtariNet(1, 4, 3)
    ens = BootstrappedAtariNet(net, 7)

    x = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)

    policies = [
        ("thompson", BootstrappedPE(ens, is_thompson=True, vote=False)),
        ("mean    ", BootstrappedPE(ens, is_thompson=False, vote=False)),
        ("vote    ", BootstrappedPE(ens, is_thompson=False, vote=True)),
    ]

    policies[0][1].sample_posterior_idx()

    for name, policy in policies:
        with torch.no_grad():
            print("\n", name, " --- ")
            print(policy.get_action(x))
