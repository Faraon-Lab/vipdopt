"""Code for Figures of Merit (FoMs)"""

from __future__ import annotations
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt

from vipdopt.utils import Number
from vipdopt.simulation import LumericalSimulation 
from vipdopt.source import Source


class FoM:
    def __init__(
            self,
            fwd_srcs: list[Source],
            adj_srcs: list[Source],
            fom_func,
            gradient_func,
            polarization: str,
            freq: Sequence[Number],
            opt_ids: Sequence[int]=None,
            **kwargs,
    ) -> None:
        self.fwd_srcs = fwd_srcs
        self.adj_srcs = adj_srcs
        self.fom_func = fom_func
        self.gradient_func = gradient_func
        self.polarization = polarization
        self.freq = freq
        self.opt_ids = list(range(len(adj_srcs))) if opt_ids is None else opt_ids
        vars(self).update(kwargs)

    def compute(self, *args, **kwargs) -> Any:
        """Compute FoM."""
        return self.fom_func(*args, **kwargs)

    def gradient(self, *args, **kwargs) -> Any:
        """Compute gradient of FoM."""
        return self.gradient_func(*args, **kwargs)


class BayerFilterFoM(FoM):

    def __init__(
            self,
            fwd_srcs: list[Source], 
            adj_srcs: list[Source],
            polarization: str,
            freq: Sequence[Number],
            opt_ids: Sequence[int]=None,
            **kwargs,
    ) -> None:
        super().__init__(
            fwd_srcs,
            adj_srcs,
            self._bayer_fom,
            self._bayer_gradient,
            polarization,
            freq,
            opt_ids,
            **kwargs,
        )
    
    def _bayer_fom(self):
        total_tfom = np.zeros(self.adj_srcs[0].shape) # FoM for transmission monitor
        total_ffom = np.zeros(self.adj_srcs[0].shape) # FoM for focal monitor
        for source in self.adj_srcs:
            T = source.transmission_magnitude()
            total_tfom += T[..., self.opt_ids]

            E = source.E()
            total_ffom += np.sum(np.square(np.abs(E[..., self.opt_ids])), axis=0)
        
        return total_tfom, total_ffom

    def _bayer_gradient(self):
        E_fwd = self.fwd_srcs[0].E()
        E_adj = self.adj_srcs[0].E()
        df_dev = np.real(np.sum(E_fwd * E_adj, axis=0))
        grad = df_dev[..., self.opt_ids]
        
        return grad