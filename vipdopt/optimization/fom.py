"""Code for Figures of Merit (FoMs)"""

from __future__ import annotations
from typing import Any, Callable, Sequence

from vipdopt.utils import Number
from vipdopt.simulation import ISimulation


class FoM:
    def __init__(
            self,
            sim: ISimulation,
            monitors: list[str],
            fom_func,
            gradieny_func,
            polarization: str,
            freq: Sequence[Number],
            opt_ids: Sequence[int]=None,
            restricted_ids: Sequence[int]=[],
            weight: float=1.0,
    ) -> None:
        self.f

    def compute(self) -> Any:
        """Compute FoM."""
        # Needs monitors for adjoint sources
        # for each adjoint source
        # Get transmission from monitor and store it
        # Then compute

        # Should only need fwd and adj sourc
        pass

    def gradient(self) -> Any:
        """Compute gradient of FoM."""
        pass

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass
    
    def __add__(self, y) -> Any:
        return self() + y()
        
