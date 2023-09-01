import numpy as np


class Filter:
    def __init__(self, bounds) -> None:
        self.bounds = bounds

    def __repr__(self) -> str:
        return f"Unknown filter with bounds: {self.bounds}"

    def verify_bounds(self, variable):
        return bool(
            (np.min(variable) >= self.bounds[0])
            and (np.max(variable) <= self.bounds[1])
        )

    def forward(self, x):
        raise NotImplementedError("Filter has no defined forward method.")

    def chain_rule(self, derivative, y, x):
        raise NotImplementedError("Filter has no defined forward method")
