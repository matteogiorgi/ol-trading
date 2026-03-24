from __future__ import annotations

import pandas as pd

from abc import ABC, abstractmethod


class Expert(ABC):
    """
    Base class for all experts.
    Each expert returns a signal in {0, 1}.
    """

    name: str = "expert"
    min_history: int = 1

    @abstractmethod
    def signal(self, history: pd.DataFrame) -> int:
        """
        Compute the signal using only past data.
        Must return 0 or 1.
        """
        raise NotImplementedError


class BuyAndHoldExpert(Expert):
    name = "buy_and_hold"
    min_history = 1

    def signal(self, history: pd.DataFrame) -> int:
        return 1


class CashExpert(Expert):
    name = "cash"
    min_history = 1

    def signal(self, history: pd.DataFrame) -> int:
        return 0


class MomentumExpert(Expert):
    def __init__(self, window: int) -> None:
        self.window = window
        self.name = f"momentum_{window}"
        self.min_history = window

    def signal(self, history: pd.DataFrame) -> int:
        recent_sum = history["return"].tail(self.window).sum()
        return 1 if recent_sum > 0 else 0


class MeanReversionExpert(Expert):
    def __init__(self, window: int) -> None:
        self.window = window
        self.name = f"mean_reversion_{window}"
        self.min_history = window

    def signal(self, history: pd.DataFrame) -> int:
        recent_sum = history["return"].tail(self.window).sum()
        return 1 if recent_sum < 0 else 0


class MovingAverageCrossoverExpert(Expert):
    def __init__(self, short_window: int, long_window: int) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be strictly smaller than long_window")

        self.short_window = short_window
        self.long_window = long_window
        self.name = f"ma_crossover_{short_window}_{long_window}"
        self.min_history = long_window

    def signal(self, history: pd.DataFrame) -> int:
        short_ma = history["price"].tail(self.short_window).mean()
        long_ma = history["price"].tail(self.long_window).mean()
        return 1 if short_ma > long_ma else 0


def build_default_experts() -> list[Expert]:
    """
    Recommended initial set of experts.
    """
    return [
        BuyAndHoldExpert(),
        CashExpert(),
        MomentumExpert(window=5),
        MomentumExpert(window=20),
        MeanReversionExpert(window=3),
        MovingAverageCrossoverExpert(short_window=5, long_window=20),
    ]
