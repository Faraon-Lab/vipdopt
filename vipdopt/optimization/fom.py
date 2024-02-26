"""Code for Figures of Merit (FoMs)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from copy import copy
from typing import Any, no_type_check

import numpy as np
import numpy.typing as npt

from vipdopt.monitor import Monitor
from vipdopt.simulation import LumericalSimulation
from vipdopt.utils import Number


class FoM:
    """Generic class for computing a figure of merit (FoM).

    Supports arithmetic operations on class instances, either with each other,
    or with scalar values.

    Attributes:
        _COUNTER (int): The current count of FoMs. Used for generating unique names.
        fom_srcs (tuple[Source]): The sources to use for computing the FoM.
        grad_srcs (tuple[Source]): The sources to use for computing the gradient.
        polarization (str): Polarization to use.
        freq (Sequence[Number]): All of the frequency bands.
        opt_ids (list[int]): List of indices specifying which frequency bands are being
            used in the optimization. Defaults to [0, ..., n_freq].
        fom_func (Callable[..., npt.ArrayLike]): The function to compute the FoM.
        grad_func (Callable[..., npt.ArrayLike]): The function to compute the gradient.
    """
    _COUNTER = 0
    def __init__(
            self,
            fom_monitors: Sequence[Monitor],
            grad_monitors: Sequence[Monitor],
            fom_func: Callable[..., npt.NDArray],
            gradient_func: Callable[..., npt.NDArray],
            polarization: str,
            freq: Sequence[Number],     # freq actually refers to the WHOLE lambda vector
            opt_ids: Sequence[int] | None=None,
            name: str='',
    ) -> None:
        """Initialize an FoM."""
        self.fom_monitors = tuple(fom_monitors)
        self.grad_monitors = tuple(grad_monitors)
        self.fom_func = fom_func
        self.gradient_func = gradient_func
        self.polarization = polarization
        self.freq = freq
        self.opt_ids = list(range(len(freq))) if opt_ids is None else opt_ids
        if name == '':
            self.name = f'fom_{FoM._COUNTER}'
        else:
            self.name = name
        FoM._COUNTER += 1

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, FoM):
            return self.fom_monitors == __value.fom_monitors and \
            self.grad_monitors == __value.grad_monitors and \
            self.fom_func.__name__ == __value.fom_func.__name__ and \
            self.gradient_func.__name__ == __value.gradient_func.__name__ and \
            self.polarization == __value.polarization and \
            self.freq == __value.freq and \
            self.opt_ids == __value.opt_ids
        return super().__eq__(__value)

    def compute(self, *args, **kwargs) -> npt.NDArray:
        """Compute FoM."""
        return self.fom_func(*args, **kwargs)

    def gradient(self, *args, **kwargs) -> npt.NDArray:
        """Compute gradient of FoM."""
        return self.gradient_func(*args, **kwargs)

    def as_dict(self) -> dict:
        """Return a dictionary representation of this FoM."""
        data = {}
        data['type'] = type(self).__name__
        data['fom_monitors'] = [
            (mon.source_name, mon.monitor_name) for mon in self.fom_monitors
        ]
        data['grad_monitors'] = [
            (mon.source_name, mon.monitor_name) for mon in self.grad_monitors
        ]
        if data['type'] == 'FoM':  # Generic FoM needs to copy functions
            data['fom_func'] = self.fom_func
            data['gradient_func'] = self.gradient_func
        data['polarization'] = self.polarization
        data['freq'] = self.freq
        data['opt_ids'] = self.opt_ids

        return data

    @staticmethod
    def from_dict(
        name: str,
        og_data: dict,
        src_to_sim_map: dict[str, LumericalSimulation],
    ) -> type[FoM]:
        """Create a figure of merit from a dictionary and list of simulations."""
        data = copy(og_data)
        fom_cls: type[FoM] = getattr(sys.modules[__name__], data.pop('type'))
        if 'weight' in data:
            del data['weight']
        data['name'] = name
        # if 'fom_func' in data:
        # if 'gradient_func' in data:

        data['fom_monitors'] = [
            # Monitor(simulation, source, monitor)
            Monitor(src_to_sim_map[src], src, mname)
            for src, mname in data['fom_monitors']
        ]
        data['grad_monitors'] = [
            Monitor(src_to_sim_map[src], src, mname)
            for src, mname in data['grad_monitors']
        ]
        return fom_cls(**data)

    @staticmethod
    @no_type_check
    def _math_helper(first: FoM | Number, second: FoM | Number, operator: str) -> FoM:
        """Helper function for arithmetic operations.

        Returns NotImplemented if operating on two FoMs and they do not have the same
        polarization, freq, and/or opt_ids.
        """
        match operator:
            case '+':
                func = np.add
            case '-':
                func = np.subtract
            case '*':
                func = np.multiply
            case '/':
                func = np.divide
            case _:
                raise NotImplementedError(f'Unrecognized operation "{operator}";'
                                 r' choose one of \{+, -, /, *\} ')


        if isinstance(second, Number):
            og_fom_func = first.fom_func
            og_grad_func = first.gradient_func
            return FoM(
                first.fom_monitors,
                first.grad_monitors,
                lambda *args, **kwargs: func(og_fom_func(*args, **kwargs), second),
                lambda *args, **kwargs: func(og_grad_func(*args, **kwargs), second),
                first.polarization,
                first.freq,
                first.opt_ids,
            )
        if isinstance(first, Number):
            og_fom_func = second.fom_func
            og_grad_func = second.gradient_func
            return FoM(
                second.fom_monitors,
                second.grad_monitors,
                lambda *args, **kwargs: func(first, og_fom_func(*args, **kwargs)),
                lambda *args, **kwargs: func(first, og_grad_func(*args, **kwargs)),
                second.polarization,
                second.freq,
                second.opt_ids,
            )

        assert isinstance(first, FoM)
        assert isinstance(second, FoM)

        # NOTE: Polarization might be unnecessary as a check
        if first.polarization != second.polarization or \
            first.opt_ids != second.opt_ids or \
                first.freq != second.freq:
            raise TypeError(
                f"unsupported operand type(s) for {operator}: 'FoM' and 'FoM' when "
                "polarization, opt_ids, and/or freq is not equal")

        og_fom_func_1 = first.fom_func
        og_grad_func_1 = first.gradient_func
        og_fom_func_2 = second.fom_func
        og_grad_func_2 = second.gradient_func

        def new_compute(*args, **kwargs):
            return func(
                og_fom_func_1(*args, **kwargs),
                og_fom_func_2(*args, **kwargs),
            )

        def new_gradient(*args, **kwargs):
            return func(
                og_grad_func_1(*args, **kwargs),
                og_grad_func_2(*args, **kwargs),
            )

        return FoM(
            first.grad_monitors + second.grad_monitors,
            first.fom_monitors + second.fom_monitors,
            new_compute,
            new_gradient,
            first.polarization,
            first.freq,
            opt_ids=first.opt_ids
        )

    def __add__(self, second: Any) -> FoM:
        """Return self + second."""
        return FoM._math_helper(self, second, '+')

    def __radd__(self, first: Any) -> FoM:
        """Return first + self."""
        return FoM._math_helper(self, first, '+')

    def __iadd__(self, second: Any) -> FoM:
        """Implement self += second."""
        combined_fom = FoM._math_helper(self, second, '+')
        self.fom_monitors = combined_fom.fom_monitors
        self.grad_monitors = combined_fom.grad_monitors
        self.fom_func = combined_fom.fom_func
        self.gradient_func = combined_fom.gradient_func
        return self

    def __sub__(self, second: Any) -> FoM:
        """Return self - second."""
        return FoM._math_helper(self, second, '-')

    def __rsub__(self, first: Any) -> FoM:
        """Return first - self."""
        return FoM._math_helper(first, self, '-')

    def __isub__(self, second: Any) -> FoM:
        """Implement self -= second."""
        combined_fom = FoM._math_helper(self, second, '-')
        vars(self).update(vars(combined_fom))
        return self

    def __mul__(self, second: Any) -> FoM:
        """Return self * second."""
        return FoM._math_helper(self, second, '*')

    def __rmul__(self, first: Any) -> FoM:
        """Return first * self."""
        return FoM._math_helper(first, self, '*')

    def __imul__(self, second: Any) -> FoM:
        """Implement self *= second."""
        combined_fom = FoM._math_helper(self, second, '*')
        vars(self).update(vars(combined_fom))
        return self

    def __truediv__(self, second: Any) -> FoM:
        """Return self / second."""
        return FoM._math_helper(self, second, '/')

    def __rtruediv__(self, first: Any) -> FoM:
        """Return first / self."""
        return FoM._math_helper(first, self, '/')

    def __itruediv__(self, second: Any) -> FoM:
        """Implement self /= second."""
        combined_fom = FoM._math_helper(self, second, '/')
        vars(self).update(vars(combined_fom))
        return self

    @staticmethod
    def zero(fom: FoM) -> FoM:
        """Return a zero FoM that is compatible for operations with `fom`.

        Returns:
            (FoM): A new FoM instance such that fom_func and grad_func are equal to 0,
                and polarization, freq, and opt_ids come from `fom`.
        """
        def zero_func(self, *args, **kwargs):  # noqa: ARG001
            return 0

        return FoM(
            [],
            [],
            zero_func,
            zero_func,
            fom.polarization,
            fom.freq,
            fom.opt_ids,
        )

    def __copy__(self) -> type[FoM]:
        """Return a copy of this FoM."""
        return self.__class__(
            self.fom_monitors,
            self.grad_monitors,
            self.fom_func,
            self.gradient_func,
            self.polarization,
            self.freq,
            self.opt_ids,
            self.name,
        )

# todo: add the following
# 		self.freq_index_negative_opt: Specify and array of frequency indices that should be optimized with a negative gradient. This is useful for making sure light does not focus to a point, for example.
# 		self.fom = np.array([]) # The more convenient way to look at fom. For example, power transmission through a monitor even if you're optimizing for a point source.
# 		self.restricted_fom = np.array([]) # FOM that is being restricted. For instance, frequencies that should not be focused.
# 		self.true_fom = np.array([]) # The true FoM being used to define the adjoint source.
# 		self.gradient = np.array([])
# 		self.restricted_gradient = np.array([])
	
# 		self.tempfile_fwd_name = ''
# 		self.tempfile_adj_name = ''
		
# 		self.design_fwd_fields = None
# 		self.design_adj_fields = None

# 		# Boolean array specifying which frequencies are active.
# 		# This is a bit confusing. Almost deprecated really. enabled means of the frequencies being optimized, which are enabled. Useful in rare circumstances where some things need to be fully disable to help catch up.
# 		self.enabled = np.ones((len(self.freq_index_opt)))
# 		# Adding this once I started optimizing for functions we DONT want (i.e. restricted_gradient). This is of the freq_index_opt_restricted values, which do we want to optimize?
# 		self.enabled_restricted = np.ones((len(self.freq_index_restricted_opt)))

class BayerFilterFoM(FoM):
    """FoM implementing the particular figure of merit for the SonyBayerFilter."""

    def __init__(
            self,
            fom_monitors: Sequence[Monitor],
            grad_monitors: Sequence[Monitor],
            polarization: str,
            freq: Sequence[Number],
            opt_ids: Sequence[int] | None=None,
            name: str='',
    ) -> None:
        """Initialize a BayerFilterFoM."""
        super().__init__(
            fom_monitors,
            grad_monitors,
            self._bayer_fom,
            self._bayer_gradient,
            polarization,
            freq,
            opt_ids,
            name,
        )

    def _bayer_fom(self):
        """Compute bayer filter figure of merit."""
        total_tfom = np.zeros(self.fom_monitors[0].shape) # FoM for transmission monitor
        total_ffom = np.zeros(self.fom_monitors[0].shape) # FoM for focal monitor
        for source in self.fom_monitors:
            transmission = source.trans_mag()
            total_tfom += transmission[..., self.opt_ids]

            efield = source.e()
            total_ffom += np.sum(np.square(np.abs(efield[..., self.opt_ids])), axis=0)

        return total_tfom, total_ffom

    def _bayer_gradient(self):
        """Compute the gradient of the bayer filter figure of merit."""
        e_fwd = self.grad_monitors[0].e()
        e_adj = self.grad_monitors[1].e()
        df_dev = np.real(np.sum(e_fwd * e_adj, axis=0))
        return df_dev[..., self.opt_ids]
