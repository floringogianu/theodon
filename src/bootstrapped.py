""" Policy Evaluation and Policy Improvement objects for Bootstrapped
    estimators.
"""

from typing import NamedTuple
from numpy import random
from gym.spaces import Discrete

import torch
from torch.nn import functional as F
from torch.distributions import MultivariateNormal


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

        self._priority = priority
        self._action_selection = action_selection
        self._single_head = (action_selection == "thompson") and not priority
        self._select_action = getattr(self, f"_{action_selection:s}_action")
        if action_selection == "thompson":
            self.episode_end_callback = self._reset_head
        else:
            self.episode_end_callback = None
        if priority:
            self._get_priority = getattr(self, f"_{priority:s}_priority")
        else:
            self._get_priority = None

        self.__head_idx = None
        self.policy = self
        self._epsilon = epsilon

        try:
            self._action_space.sample()
        except AttributeError:
            self._action_space = Discrete(self._action_space)

        acts_no = self._action_space.n
        self.__actions_cuda_range = (
            torch.linspace(0, acts_no - 1, acts_no)
            .long()
            .view(1, 1, acts_no)
            .to("cuda")
        )

    def get_action(self, state):
        q_value = None
        priority = None
        q_values = None

        if state.size(0) > 1:
            raise RuntimeError("No batches yet.")

        with torch.no_grad():
            epsilon = next(self._epsilon)
            if random.uniform() > epsilon:
                if self._single_head:  # we don't care about about all q_values
                    if self.__head_idx is None:
                        raise RuntimeError("You did not reset the head.")
                    q_values = self._estimator(state, head_idx=self.__head_idx)
                    q_values = q_values.unsqueeze(1)
                else:
                    q_values = self._estimator(state)
                action = self._select_action(q_values).item()
            else:
                action = random.randint(0, self._action_space.n)
                if self._get_priority is not None:
                    q_values = self._estimator(state)
                    priority = self._get_priority(q_values)

            if q_values is not None:
                q_value = q_values.select(-1, action).mean(dim=-1).item()

            if self._get_priority:
                priority = self._get_priority(q_values)

        return DeterministicEnsembleOutput(
            action=action,
            priority=priority,
            q_value=q_value,
            head_idx=self.__head_idx,
        )

    def _best_mean_action(self, q_values):
        """ Also used with thompson when there would be a single head here.
        """
        if q_values.ndimension() != 3:
            raise ValueError("q_values is supposed to be 3D.")
        q_values = q_values.mean(dim=1)
        _, actions = q_values.max(dim=1)
        return actions

    def _sample_best_action(self, q_values):
        """ Samples one of the proposed best actions.
        """
        if q_values.ndimension() != 3:
            raise ValueError("q_values is supposed to be 3D.")
        batch_size, heads_no, _actions_no = q_values.shape
        proposals = q_values.argmax(dim=2)
        sampled_heads = torch.randint(0, heads_no, (batch_size, 1))
        actions = proposals.gather(1, sampled_heads)
        return actions

    def _most_voted_action(self, q_values):
        """ Samples one of the proposed best actions.
        """
        if q_values.ndimension() != 3:
            raise ValueError("q_values is supposed to be 3D.")
        bests = q_values.argmax(dim=2)
        nvotes = bests.unsqueeze(-1).expand(-1, -1, self._action_space.n)
        votes = nvotes.eq(self.__actions_cuda_range).sum(dim=1)
        max_counts, _ = votes.max(dim=1)
        batch_probs = (votes == max_counts.unsqueeze(1)).float().cpu().numpy()
        batch_probs = batch_probs / batch_probs.sum(axis=1, keepdims=True)

        actions = []
        for probs in batch_probs:
            actions.append(random.choice(range(self._action_space.n), p=probs))

        return torch.tensor(actions, dtype=torch.long)

    def _best_from_gaussian_sample_action(self, q_values):
        if q_values.ndimension() != 3:
            raise ValueError("q_values is supposed to be 3D.")
        _, heads_no, _actions_no = q_values.shape
        means = q_values.mean(dim=1)
        centered_qs = q_values - means.unsqueeze(1)
        covs = torch.bmm(centered_qs.transpose(1, 2), centered_qs) / heads_no
        dist = MultivariateNormal(means, covs)

        alpha = 1e-4
        while True:
            try:        
                dist = MultivariateNormal(means, covs)
                break
            except RuntimeError:
                covs += alpha * torch.eye(heads_no).unsqueeze(0)
                alpha *= 10
        actions = dist.sample((1,)).argmax(dim=2)
        return actions.squeeze(1)


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

    @property
    def action_selection(self) -> str:
        return self._action_selection

    @action_selection.setter
    def action_selection(self, action_selection: str) -> None:
        self._action_selection = action_selection
        priority = self._priority

        self._single_head = (action_selection == "thompson") and not priority
        self._select_action = getattr(self, f"_{action_selection:s}_action")
        if action_selection == "thompson":
            self.episode_end_callback = self._reset_head
        else:
            self.episode_end_callback = None


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
