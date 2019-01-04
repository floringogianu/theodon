""" Policy Evaluation and Policy Improvement objects for Bootstrapped
    estimators.
"""
from functools import partial
import torch
from wintermute.policy_improvement import DQNPolicyImprovement, get_dqn_loss


class BootstrappedEpsilonGreedyPolicy:
    """ TODO: implement this.
    """
    pass


class BootstrappedDQNPolicyImprovement(DQNPolicyImprovement):
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
        boot_mask = None
        batch = [el.to(self.device) for el in batch]

        if len(batch) == 6:
            boot_mask = batch[-1]
            batch = batch[:5]

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
        elif boot_mask is not None:
            # split batch in mini-batches for each ensemble component.
            idxs = boot_mask.unique()
            batches = [[el[boot_mask == idx] for el in batch] for idx in idxs]

            # Gather the losses for each batch and ensemble component. We use
            # partial application to set which ensemble component gets trained.
            dqn_losses = [
                get_dqn_loss(
                    batch_,
                    partial(self.estimator, mid=idx),
                    self.gamma,
                    target_estimator=partial(self.target_estimator, mid=idx),
                    is_double=self.is_double,
                ).loss
                for idx, batch_ in zip(idxs, batches)
            ]

            # recompose the dqn_loss
            dqn_loss = torch.zeros((batch_sz, 1), device=dqn_losses[0].device)
            for idx, component_loss in zip(idxs, dqn_losses):
                dqn_loss[boot_mask == idx] = component_loss

        # scenario 3: all ensemble components, all data
        elif boot_mask is None:
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

    @posterior_idx.setter
    def posterior_idx(self, idx):
        assert self.__is_thompson, (
            "Calling this setter means you want to train the ensemble in a "
            + "Thompson sampling setup but you didn't set `is_thompson` in the "
            + "constructor."
        )
        self.__posterior_idx = idx
