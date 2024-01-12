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
            fom_srcs: Sequence[Source],
            grad_srcs: Sequence[Source],
            fom_func,
            gradient_func,
            polarization: str,
            freq: Sequence[Number],
            opt_ids: Sequence[int]=None,
            **kwargs,
    ) -> None:
        self.fom_srcs = tuple(fom_srcs)
        self.grad_srcs = tuple(grad_srcs)
        self.fom_func = fom_func
        self.gradient_func = gradient_func
        self.polarization = polarization
        self.freq = freq
        self.opt_ids = list(range(len(freq))) if opt_ids is None else opt_ids
        vars(self).update(kwargs)

    def compute(self, *args, **kwargs) -> npt.ArrayLike:
        """Compute FoM."""
        return self.fom_func(*args, **kwargs)

    def gradient(self, *args, **kwargs) -> npt.ArrayLike:
        """Compute gradient of FoM."""
        return self.gradient_func(*args, **kwargs)
    
    def _math_helper(first: FoM | Number, second: FoM | Number, operator: str) -> FoM:
        match operator:
            case '+':
                func = np.add
            case '-':
                func = np.subtract
            case '*':
                func = np.multiply
            case '/':
                func = np.divide

        if isinstance(second, Number):
            return FoM(
                first.fom_srcs,
                first.grad_srcs,
                lambda *args, **kwargs: func(first.compute(*args, **kwargs), second),
                lambda *args, **kwargs: func(first.gradient(*args, **kwargs), second),
                first.polarization,
                first.freq,
                first.opt_ids,
            )
        elif isinstance(first, Number):
            return FoM(
                second.fom_srcs,
                second.grad_srcs,
                lambda *args, **kwargs: func(first, second.compute(*args, **kwargs)),
                lambda *args, **kwargs: func(first, second.gradient(*args, **kwargs)),
                second.polarization,
                second.freq,
                second.opt_ids,
            )
        
        assert isinstance(first, FoM)
        assert isinstance(second, FoM)

        if first.polarization != second.polarization:
            return NotImplemented
        if first.opt_ids!= second.opt_ids:
            return NotImplemented
        if first.freq != second.freq:
            return NotImplemented

        def new_compute(*args, **kwargs):
            return func(
                first.compute(*args, **kwargs),
                second.compute(*args, **kwargs),
            )
        
        def new_gradient(*args, **kwargs):
            return func(
                first.gradient(*args, **kwargs),
                second.gradient(*args, **kwargs),
            )
        
        return FoM(
            first.grad_srcs + second.grad_srcs,
            first.fom_srcs + second.fom_srcs,
            new_compute,
            new_gradient,
            first.polarization,
            first.freq,
            opt_ids=first.opt_ids
        )
    
    def __add__(self, second: Any) -> FoM:
        return FoM._math_helper(self, second, '+')
    
    def __radd__(self, first: Any) -> FoM:
        return FoM._math_helper(self, first, '+')

    def __iadd__(self, second: Any) -> FoM:
        combined_FoM = FoM._math_helper(self, second, '+')
        vars(self).update(vars(combined_FoM))
        return self

    def __sub__(self, second: Any) -> FoM:
        return FoM._math_helper(self, second, '-')
    
    def __rsub__(self, first: Any) -> FoM:
        return FoM._math_helper(first, self, '-')

    def __isub__(self, second: Any) -> FoM:
        combined_FoM = FoM._math_helper(self, second, '-')
        vars(self).update(vars(combined_FoM))
        return self

    def __mul__(self, second: Any) -> FoM:
        return FoM._math_helper(self, second, '*')
    
    def __rmul__(self, first: Any) -> FoM:
        return FoM._math_helper(first, self, '*')

    def __imul__(self, second: Any) -> FoM:
        combined_FoM = FoM._math_helper(self, second, '*')
        vars(self).update(vars(combined_FoM))
        return self

    def __truediv__(self, second: Any) -> FoM:
        return FoM._math_helper(self, second, '/')

    def __rtruediv__(self, first: Any) -> FoM:
        return FoM._math_helper(first, self, '/')

    def __itruediv__(self, second: Any) -> FoM:
        combined_FoM = FoM._math_helper(self, second, '/')
        vars(self).update(vars(combined_FoM))
        return self

FOM_ZERO = FoM([], [], lambda *args, **kwargs: 0, lambda *args, **kwargs: 0, '', [])
    

class BayerFilterFoM(FoM):

    def __init__(
            self,
            fom_srcs: Sequence[Source], 
            grad_srcs: Sequence[Source],
            polarization: str,
            freq: Sequence[Number],
            opt_ids: Sequence[int]=None,
            **kwargs,
    ) -> None:
        super().__init__(
            fom_srcs,
            grad_srcs,
            self._bayer_fom,
            self._bayer_gradient,
            polarization,
            freq,
            opt_ids,
            **kwargs,
        )
    
    def _bayer_fom(self):
        total_tfom = np.zeros(self.fom_srcs[0].shape) # FoM for transmission monitor
        total_ffom = np.zeros(self.fom_srcs[0].shape) # FoM for focal monitor
        for source in self.fom_srcs:
            T = source.transmission_magnitude()
            total_tfom += T[..., self.opt_ids]

            E = source.E()
            total_ffom += np.sum(np.square(np.abs(E[..., self.opt_ids])), axis=0)
        
        return total_tfom, total_ffom

    def _bayer_gradient(self):
        E_fwd = self.grad_srcs[0].E()
        E_adj = self.grad_srcs[1].E()
        df_dev = np.real(np.sum(E_fwd * E_adj, axis=0))
        grad = df_dev[..., self.opt_ids]
        
        return grad