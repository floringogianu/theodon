""" Policy Evaluation and Policy Improvement objects for Bootstrapped
    estimators.
"""
from functools import partial
from typing import NamedTuple

import torch
from gym.spaces import Discrete
from termcolor import colored as clr

from wintermute.policy_improvement import DQNPolicyImprovement, get_dqn_loss
from wintermute.policy_improvement import DQNLoss


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

    def __init__(self, estimator, is_thompson=False, vote=True, test=False):
        self.__estimator = estimator
        self.__posterior_idx = None
        self.__vote = vote
        self.__ensemble_sz = len(self.__estimator)
        params = estimator.parameters()
        try:
            self.__device = next(params).device
        except TypeError:
            self.__device = next(params[0]["params"]).device

        # TODO: Most likely this policy evaluation step is not woring
        # with batches. Need to test.

        cls_name = f"{self.__class__.__name__}"
        if is_thompson:
            self.get_action = self.__thompson_action
            print(clr(f"{cls_name} with Thompson Sampling.", "green"))
        elif vote:
            print(clr(f"{cls_name} with Voting Ensemble.", "green"))
            self.get_action = self.__voting_action
        else:
            print(clr(f"{cls_name} with Plain Ensemble.", "green"))
            self.get_action = self.__mean_action

        if test:
            if vote:
                print(clr(f"{cls_name} with Voting Ensemble.", "magenta"))
                self.get_action = self.__voting_action
            else:
                print(clr(f"{cls_name} with Plain Ensemble.", "green"))
                self.get_action = self.__mean_action

    def __thompson_action(self, state):
        assert (
            self.__posterior_idx is not None
        ), "Call `sample_posterior_idx` first."

        state = state.to(self.__device)
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
        state = state.to(self.__device)
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
        state = state.to(self.__device)
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

    @property
    def estimator(self):
        return self.__estimator


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
        boot_mask=True,
    ):
        super().__init__(
            estimator, optimizer, gamma, target_estimator, is_double
        )
        self.__is_thompson = is_thompson
        self.__posterior_idx = None

        assert (
            boot_mask != is_thompson
        ), "Can't have both masks and Thompson Sampling, yet."

        cls_name = f"{self.__class__.__name__}"
        if self.__is_thompson:
            self.__get_dqn_loss = self.__thompson_update
            print(clr(f"{cls_name} with Thompson Sampling.", "green"))
        elif boot_mask:
            self.__get_dqn_loss = self.__bootstrapp_update
            print(clr(f"{cls_name} with Data Bootstrapping.", "green"))
        elif not boot_mask:
            self.__get_dqn_loss = self.__ensemble_update
            print(clr(f"{cls_name} with Plain Ensembles.", "green"))
        else:
            raise NotImplementedError("Not sure what to do...")

    def __call__(self, batch, cb=None):
        """ Overwrites the parent's methods.
        """
        dqn_loss = self.__get_dqn_loss(batch)

        if cb:
            loss = cb(dqn_loss)
        else:
            loss = dqn_loss.loss.mean()

        loss.backward()
        self.update_estimator()

    def set_posterior_idx(self, idx):
        assert self.__is_thompson, (
            "Calling this setter means you want to train the ensemble in a "
            + "Thompson sampling setup but you didn't set `is_thompson` in the "
            + "constructor."
        )
        self.__posterior_idx = idx

    def get_uncertainty(self, x, actions, cached_features=True):
        """ Returns the predictive uncertainty of the model.

        Args:
            x (torch.tensor): Can be either a batch of states or of features.
            actions (torch.tensor): The actions we need uncertainties for.
            cached_features (bool, optional): Defaults to True. If True then
                `x` is a batch of features.

        Returns:
            torch.tensor: A vector of predictive variances.
        """

        with torch.no_grad():
            ensemble_qvals = self.estimator(x, cached_features=cached_features)
            qvals_var = ensemble_qvals.var(0).gather(1, actions)
        return qvals_var

    def __thompson_update(self, batch):
        # scenario 1: sampled component, all data
        batch = [el.to(self.device) for el in batch]
        idx = self.__posterior_idx
        loss = get_dqn_loss(
            batch,
            partial(self.estimator, mid=idx),
            self.gamma,
            target_estimator=partial(self.target_estimator, mid=idx),
            is_double=self.is_double,
        )

        return DQNLoss(
            loss=loss.loss,
            priority=self.get_uncertainty(
                batch[0], batch[1], cached_features=False
            ),
            q_values=None,
            q_targets=None,
        )

    def __bootstrapp_update(self, batch):
        # scenario 2: all ensemble components, masked data
        batch, boot_masks = batch
        bsz = batch[0].shape[0]

        batch = [el.to(self.device) for el in batch]
        boot_masks.to(self.device)

        # pass through the feature extractor and replace states
        # with features. Also pass next_states once more if Double-DQN.
        if self.estimator.feature_extractor is not None:
            online = self.estimator.get_features
            target = self.target_estimator.get_features
            batch[0] = online(batch[0])
            if self.is_double:
                with torch.no_grad():
                    features_ = online(batch[3])
            batch[3] = target(batch[3])

        # split batch in mini-batches for each ensemble component.
        # because now state and state_ have differen dimensions we cannot do:
        # batches = [[el[bm] for el in batch] for bm in boot_masks]
        # instead we mask the bootmask too... :(
        batches = []
        for bm in boot_masks:
            batches.append(
                [
                    [
                        batch[0][bm],
                        batch[1][bm],
                        batch[2][bm],
                        batch[3][bm[batch[4].squeeze()]],
                        batch[4][bm],
                    ],
                    bm[batch[4].squeeze()],
                ]
            )

        # Gather the losses for each batch and ensemble component. We use
        # partial application to set which ensemble component gets trained.
        dqn_losses = [
            get_dqn_loss(
                batch_,
                partial(self.estimator, mid=mid, cached_features=True),
                self.gamma,
                target_estimator=partial(
                    self.target_estimator, mid=mid, cached_features=True
                ),
                is_double=self.is_double,
                next_states_features=features_[bm] if self.is_double else None,
            ).loss
            for mid, (batch_, bm) in enumerate(batches)
        ]

        # sum up the losses of a given transition across ensemble components
        dqn_loss = torch.zeros((bsz, 1), device=dqn_losses[0].device)
        for loss, boot_mask in zip(dqn_losses, boot_masks):
            dqn_loss[boot_mask] += loss

        # TODO: gradient rescalling!!!

        return DQNLoss(
            loss=dqn_loss,
            priority=self.get_uncertainty(
                batch[0], batch[1], cached_features=True
            ),
            q_values=None,
            q_targets=None,
        )

    def __ensemble_update(self, batch):
        # scenario 3: all ensemble components, all data
        batch = [el.to(self.device) for el in batch]
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
        loss = torch.cat(dqn_losses, dim=1).mean(dim=1).unsqueeze(1)

        return DQNLoss(
            loss=loss,
            priority=self.get_uncertainty(
                batch[0], batch[1], cached_features=False
            ),
            q_values=None,
            q_targets=None,
        )


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
