""" Policy Evaluation and Policy Improvement objects for Bootstrapped
    estimators.
"""
from functools import partial
from typing import NamedTuple
from numpy import random

import torch
from termcolor import colored as clr

from wintermute.policy_improvement import DQNPolicyImprovement, get_dqn_loss
from wintermute.policy_improvement import DQNLoss
from wintermute.policy_evaluation import get_epsilon_schedule


class DeterministicEnsembleOutput(NamedTuple):
    """ The output of the deterministic policy. """

    action: int
    q_value: float
    priority: object
    full: object
    posterior: object
    vote_cnt: object


def selected_action_variance(ensemble_qvals, act, _votes=None):
    """ Returns the variance of the predicted values for the selected action.
    """
    if ensemble_qvals.ndimension() == 2:
        return ensemble_qvals[:, act].var().item()
    elif ensemble_qvals.ndimension() == 3:
        return ensemble_qvals.var(0).gather(1, act)
    raise ValueError("Unexepected number of dimensions")


def best_action_variance(ensemble_qvals, _action, votes=None):
    """ This needs further study. Currently p * (1 - p) with p=P(best_action)
    """
    if ensemble_qvals.ndimension() == 2:
        # Here ensemble_qvals is shaped (ncomponents, nactions)
        if votes is None:
            nactions = ensemble_qvals.size(-1)
            bests = ensemble_qvals.argmax(1)
            votes = torch.zeros(
                nactions, dtype=bests.dtype, device=bests.device
            )
            votes.put_(bests, torch.ones_like(bests), accumulate=True)
        prob = votes.max().float().item() / votes.sum().float().item()
        return prob * (1 - prob)
    elif ensemble_qvals.ndimension() == 3:
        # Here ensemble_qvals is shaped (ncomponents, batch_size, nactions)
        if votes is None:
            nmodels, batch_size, nactions = ensemble_qvals.size()
            argmaxs = ensemble_qvals.argmax(2)
            offset = (
                torch.linspace(
                    0,
                    (batch_size - 1) * nactions,
                    batch_size,
                    device=argmaxs.device,
                )
                .long()
                .unsqueeze(0)
            )
            argmaxs = (offset + argmaxs).view(-1)
            votes = torch.zeros(
                batch_size * nactions, dtype=torch.long, device=argmaxs.device
            )
            votes.put_(argmaxs, torch.ones_like(argmaxs), accumulate=True)
            votes = votes.view(batch_size, nactions)
        elif votes.ndimension() != 2:
            raise ValueError("Expected batch_size x nactions tensor")
        probs = votes.max(dim=1)[0].float() / votes.sum(dim=1).float()
        return probs * (1 - probs)
    raise ValueError("Unexepected number of dimensions")


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

    def __init__(
        self,
        estimator,
        action_no,
        epsilon,
        is_thompson=False,
        vote=True,
        test=False,
        var_method="selected_action",
    ):
        self.__estimator = estimator
        self.__posterior_idx = None
        self.__vote = vote
        self.__ensemble_sz = len(self.__estimator)
        self.policy = self
        params = estimator.parameters()
        try:
            self.__device = next(params).device
        except TypeError:
            self.__device = next(params[0]["params"]).device

        self.action_no = action_no
        self.epsilon = epsilon
        try:
            epsilon = next(self.epsilon)  # TODO: se consuma un eps aiurea :)
        except TypeError:
            self.epsilon = get_epsilon_schedule(**self.epsilon)
            epsilon = next(self.epsilon)

        # TODO: Most likely this policy evaluation step is not working
        # with batches. Need to test.

        self._get_variance = eval(var_method + "_variance")

        cls_name = f"{self.__class__.__name__}"
        if is_thompson:
            if test:
                raise ValueError("No thompson at eval time.")
            self.get_action = self.__thompson_action
            print(clr(f"{cls_name} with Thompson Sampling.", "green"))
        elif vote:
            print(clr(f"{cls_name} with Voting Ensemble.", "green"))
            self.get_action = self.__voting_action
        else:
            print(clr(f"{cls_name} with Plain Ensemble.", "green"))
            self.get_action = self.__mean_action

    def __thompson_action(self, state):
        if self.__posterior_idx is None:
            raise ValueError("Call `sample_posterior_idx` first.")

        # TODO: we might not need q_values from all heads when thompson
        state = state.to(self.__device)
        ensemble_qvals = self.__estimator(state).squeeze(1)
        if ensemble_qvals.ndimension() != 2:
            raise ValueError("It should be ncomponents x nactions")
        component_idx = self.__posterior_idx
        qvals = ensemble_qvals[component_idx]

        epsilon = next(self.epsilon)
        if epsilon > random.uniform():
            action = random.randint(0, self.action_no)
            qval = qvals[action]
        else:
            qval, argmax_a = qvals.max(0)
            action = argmax_a.item()

        variance = self._get_variance(ensemble_qvals, action)

        return DeterministicEnsembleOutput(
            action=action,
            priority=variance,
            q_value=qval.item(),
            full=ensemble_qvals,
            posterior=component_idx,
            vote_cnt=None,
        )

    def __mean_action(self, state):
        state = state.to(self.__device)
        ensemble_qvals = self.__estimator(state).squeeze(1)
        qvals = ensemble_qvals.mean(0)

        epsilon = next(self.epsilon)
        if epsilon > random.uniform():
            action = random.randint(0, self.action_no)
            qval = qvals[action]
        else:
            qval, argmax_a = qvals.max(0)
            action = argmax_a.item()

        variance = self._get_variance(ensemble_qvals, action)

        return DeterministicEnsembleOutput(
            action=action,
            priority=variance,
            q_value=qval.item(),
            full=ensemble_qvals,
            posterior=None,
            vote_cnt=None,
        )

    def __voting_action(self, state):
        state = state.to(self.__device)
        ensemble_qvals = self.__estimator(state).squeeze(1)
        act_no = ensemble_qvals.shape[1]
        qvals, argmaxs = ensemble_qvals.max(1)
        votes = torch.zeros(act_no, dtype=argmaxs.dtype, device=argmaxs.device)
        votes.put_(argmaxs, torch.ones_like(argmaxs), accumulate=True)

        epsilon = next(self.epsilon)
        if epsilon > random.uniform():
            action = random.randint(0, self.action_no)
            qval = ensemble_qvals[:, action].mean()
        else:
            action = votes.argmax().item()
            qval = qvals[argmaxs == action].mean()

        variance = self._get_variance(ensemble_qvals, action, votes)

        return DeterministicEnsembleOutput(
            action=action,
            priority=variance,
            q_value=qval,
            full=ensemble_qvals,
            posterior=None,
            vote_cnt=votes,
        )

    def sample_posterior_idx(self):
        """ Samples one of the components of the ensemble and returns its
            index.
        """
        self.__posterior_idx = torch.randint(0, self.__ensemble_sz, (1,)).item()
        return self.__posterior_idx

    def __call__(self, state):
        return self.get_action(state)

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
        var_method="selected_action",
    ):
        super().__init__(
            estimator, optimizer, gamma, target_estimator, is_double
        )
        self.__is_thompson = is_thompson
        self.__posterior_idx = None

        self._get_variance = eval(var_method + "_variance")

        if is_thompson and boot_mask:
            raise ValueError("Can't have both masks and Thompson Sampling, yet")

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

    def get_uncertainty(self, x, actions, are_features=True):
        """ Returns the predictive uncertainty of the model.

        Args:
            x (torch.tensor): Can be either a batch of states or of features.
            actions (torch.tensor): The actions we need uncertainties for.
            are_features (bool, optional): Defaults to True. If True then
                `x` is a batch of features.

        Returns:
            torch.tensor: A vector of predictive variances.
        """

        with torch.no_grad():
            ensemble_qvals = self.estimator(x, are_features=are_features)
            self._get_variance(ensemble_qvals, actions)
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
            priorities=self.get_uncertainty(
                batch[0], batch[1], are_features=False
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
        for mid, bmask in enumerate(boot_masks):
            if bmask.sum() > 0:
                batches.append(
                    (
                        mid,
                        [
                            batch[0][bmask],
                            batch[1][bmask],
                            batch[2][bmask],
                            batch[3][bmask[batch[4].squeeze()]],
                            batch[4][bmask],
                        ],
                        bmask[batch[4].squeeze()],
                    )
                )

        # Gather the losses for each batch and ensemble component. We use
        # partial application to set which ensemble component gets trained.
        dqn_losses = [
            get_dqn_loss(
                batch_,
                partial(self.estimator, mid=mid, are_features=True),
                self.gamma,
                target_estimator=partial(
                    self.target_estimator, mid=mid, are_features=True
                ),
                is_double=self.is_double,
                next_states_features=features_[next_state_mask]
                if self.is_double
                else None,
            ).loss
            for mid, batch_, next_state_mask in batches
        ]

        # sum up the losses of a given transition across ensemble components
        dqn_loss = torch.zeros((bsz, 1), device=dqn_losses[0].device)
        for loss, (mid, _, _) in zip(dqn_losses, batches):
            dqn_loss[boot_masks[mid]] += loss

        # TODO: gradient rescalling!!!

        return DQNLoss(
            loss=dqn_loss,
            priorities=self.get_uncertainty(
                batch[0], batch[1], are_features=True
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
            priorities=self.get_uncertainty(
                batch[0], batch[1], are_features=False
            ),
            q_values=None,
            q_targets=None,
        )


def __test():
    from wintermute.estimators import AtariNet, BootstrappedAtariNet

    net = AtariNet(1, 4, 3)
    ens = BootstrappedAtariNet(net, 7)

    x = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)

    policies = [
        (
            "thompson",
            BootstrappedPE(ens, 3, 0.01, is_thompson=True, vote=False),
        ),
        (
            "mean    ",
            BootstrappedPE(ens, 3, 0.01, is_thompson=False, vote=False),
        ),
        (
            "vote    ",
            BootstrappedPE(ens, 3, 0.01, is_thompson=False, vote=True),
        ),
    ]

    policies[0][1].sample_posterior_idx()

    for name, policy in policies:
        with torch.no_grad():
            print("\n", name, " --- ")
            print(policy.get_action(x))


if __name__ == "__main__":
    __test()
