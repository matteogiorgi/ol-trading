from __future__ import annotations

import numpy as np


class HedgeLearner:
    """
    Hedge / Multiplicative Weights learner.

    weights are updated as:
        w_i <- w_i * exp(eta * reward_i)
        then normalized
    """

    def __init__(self, eta: float, n_experts: int) -> None:
        if eta <= 0:
            raise ValueError("eta must be positive")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")

        self.eta = eta
        self.n_experts = n_experts
        self.weights = np.full(n_experts, 1.0 / n_experts, dtype=float)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def update(self, expert_returns: np.ndarray) -> np.ndarray:
        """
        Update weights based on expert returns.
        Returns the updated weights.
        """
        if expert_returns.shape != (self.n_experts,):
            raise ValueError(
                f"expert_returns must have shape {(self.n_experts,)}, "
                f"got {expert_returns.shape}"
            )

        new_weights = self.weights * np.exp(self.eta * expert_returns)
        total = new_weights.sum()

        if total <= 0 or not np.isfinite(total):
            raise FloatingPointError("Numerical instability in weight update")

        self.weights = new_weights / total
        return self.get_weights()
