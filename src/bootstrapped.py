""" Policy Evaluation and Policy Improvement objects for Bootstrapped
    estimators.
"""

from typing import NamedTuple
from numpy import random
from gym.spaces import Discrete

import torch
from torch.nn import functional as F


class DeterministicEnsembleOutput(NamedTuple):
    """ The output of the deterministic policy. """

    action: int
    q_value: float
    priority: object
    head_idx: int


class BootstrappedPE:
    def __init__(  # pylint: disable=bad-continuation
        self,
        estimator,
        action_space,
        epsilon,
        action_selection="thomposon",
        priority=False,
        **_kwargs,
    ):

        self._thompson_action = self._best_mean_action

        self._estimator = estimator
        self._action_space = action_space
        self._single_head = (action_selection == "thompson") and not priority
        self._select_action = getattr(self, f"_{action_selection:s}_action")
        if priority:
            self._get_priority = getattr(self, f"_{priority:s}_priority")
        else:
            self._get_priority = None
        self.__head_idx = None
        self.policy = self
        self._epsilon = epsilon

        if action_selection == "thompson":
            self.episode_end_callback = self._reset_head
        else:
            self.episode_end_callback = None

        try:
            self._action_space.sample()
        except AttributeError:
            self._action_space = Discrete(self._action_space)

    def get_action(self, state):
        q_value = None
        priority = None

        if state.size(0) > 1:
            raise RuntimeError("No batches yet.")

        with torch.no_grad():
            epsilon = next(self._epsilon)
            if random.uniform() > epsilon:
                if self._single_head:
                    if self.__head_idx is None:
                        raise RuntimeError("You did not reset the head.")
                    q_values = self._estimator(state, head_idx=self.__head_idx)
                else:
                    q_values = self._estimator(state)
                action, q_value = self._select_action(q_values)
            else:
                action = random.randint(0, self._action_space.n)
                if self._get_priority is not None:
                    q_values = self._estimator(state)
                    _, q_value = self._select_action(q_values, action=action)

            if self._get_priority:
                priority = self._get_priority(q_values)

        return DeterministicEnsembleOutput(
            action=action,
            priority=priority,
            q_value=q_value,
            head_idx=self.__head_idx,
        )

    def _best_mean_action(self, q_values, action=None) -> tuple:
        if action is None:
            if q_values.ndimension() == 3:
                q_values = q_values.mean(dim=1)
            q_value, action = q_values.max(dim=1)
            action, q_value = action.item(), q_value.item()
        else:
            q_value = q_values.select(-1, action).mean().item()
        return action, q_value

    def _reset_head(self):
        """ Samples one of the components of the ensemble and returns its
            index.
        """
        self.__head_idx = random.randint(0, self._estimator.heads_no)
        return self.__head_idx

    def __call__(self, state):
        return self.get_action(state)

    @property
    def estimator(self):
        return self._estimator


class BootstrappedPI:
    def __init__(
        self, estimator, optimizer, target, double, gamma=0.99, **_kwargs
    ):
        self._estimator = estimator
        self._target = target
        self._optimizer = optimizer

        self._double = double
        self._gamma = gamma

    def _update(self, batch):
        if len(batch) == 2:
            (states, actions, rewards, next_states, notdone), mask = batch
        else:
            (states, actions, rewards, next_states, notdone), mask = batch, None

        estimator, target = self._estimator, self._target

        q_values = estimator(states)

        qsa = q_values.gather(
            2, actions.unsqueeze(2).expand(-1, estimator.heads_no, 1)
        )

        with torch.no_grad():
            q_targets = target(next_states)
            qsa_targets = rewards.repeat(1, estimator.heads_no)
            if self._double:
                next_q_values = estimator(next_states)
                _, double_actions = next_q_values.max(dim=2)
                next_qs = q_targets.gather(2, double_actions.unsqueeze(2))
                next_qs = next_qs.squeeze(2)
            else:
                next_qs, _ = q_targets.max(dim=2)

            qsa_targets[notdone.squeeze(1)] += self._gamma * next_qs

        losses = F.smooth_l1_loss(qsa.squeeze(2), qsa_targets, reduction="none")
        if mask is not None:
            losses = losses[mask]
        self._optimizer.zero_grad()
        losses.mean().backward()
        self._optimizer.step()
        torch.cuda.synchronize()

    def update_target_estimator(self):
        self._target.load_state_dict(self._estimator.state_dict())

    def __call__(self, batch, cb=None):
        self._update(batch)

    @property
    def estimator(self):
        return self._estimator
