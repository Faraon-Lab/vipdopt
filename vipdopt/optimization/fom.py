"""Code for Figures of Merit (FoMs)."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import copy
from functools import reduce
from itertools import product, starmap
from typing import Any, no_type_check

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt.simulation import LumericalSimulation, Monitor, Source
from vipdopt.utils import Number, flatten, starmap_with_kwargs

POLARIZATIONS = ['TE', 'TM', 'TE+TM']


class SuperFoM:
    """Representation of a weighted sum of FoMs that take the same arguments."""

    def __init__(self, foms: list[Sequence[FoM2]], weights: Sequence[float] = (1.0,)) -> None:
        """Initialize a SuperFoM."""
        self.foms: list[tuple[FoM2, ...]] = [tuple(f) for f in foms]
        self.weights: list[float] = list(weights)

    def __copy__(self) -> SuperFoM:
        """Create a copy of this FoM."""
        return SuperFoM(self.foms, self.weights)

    def __eq__(self, other: Any) -> bool:
        """Test equality."""
        if isinstance(other, SuperFoM):
            return self.foms == other.foms and self.weights == other.weights
        return super().__eq__(other)

    def compute_fom(self, *args, **kwargs) -> npt.NDArray:
        """Compute the weighted sum of the FoMs."""
        fom_results = np.array(
            list(
                SuperFoM._compute_prod(
                    FoM2.compute_fom,
                    fom_tup,
                    *args,
                    **kwargs,
                ) for fom_tup in self.foms
            )
        )
        return np.einsum('i,i...->...', self.weights, fom_results)

    def _compute_prod(function: Callable, foms: tuple[FoM2, ...], *args, **kwargs) -> npt.NDArray:
        factors = np.array(list(
            starmap_with_kwargs(
                function,
                ((fom, *args) for fom in foms),
                (kwargs for _ in foms)
            )
        ))
        return np.prod(factors, axis=0)

    def compute_grad(self, *args, **kwargs) -> npt.NDArray:
        """Compute the weighted sum of the gradients."""
        grad_results = np.array(
            list(
                SuperFoM._compute_prod(
                    FoM2.compute_grad,
                    fom_tup,
                    *args,
                    **kwargs,
                ) for fom_tup in self.foms
            )
        )
        return np.einsum('i,i...->...', self.weights, grad_results)

    def create_forward_sim(self, base_sim: LumericalSimulation) -> list[LumericalSimulation]:
        """Create all unique forward simulations needed to compute this FoM."""
        fwd_sim_map = unique_fwd_sim_map(self.foms)
        return [base_sim.with_enabled(srcs) for srcs in fwd_sim_map]

    def create_adjoint_sim(self, base_sim: LumericalSimulation) -> list[LumericalSimulation]:
        """Create all unique adjoint simulations needed to compute this FoM."""
        adj_sim_map = unique_adj_sim_map(self.foms)
        return [base_sim.with_enabled(srcs) for srcs in adj_sim_map]

    def _math_helper(first: SuperFoM | Number, second: SuperFoM | Number, operator: str) -> SuperFoM:
        """Helper function for computing arithmetic functions."""
        foms = []
        weights = []
        if isinstance(first, SuperFoM) and isinstance(second, SuperFoM):
            match operator:
                case '+':
                    foms = first.foms + second.foms
                    weights = first.weights + second.weights
                case '-':
                    foms = first.foms + second.foms
                    weights = first.weights + [-w for w in second.weights]
                case '*':
                    foms = list(flatten(tup) for tup in product(first.foms, second.foms))
                    weights = [reduce(lambda x, y: x * y, tup) for tup in product(first.weights, second.weights)]
                case _:
                    raise NotImplementedError
        elif isinstance(first, SuperFoM) and isinstance(second, Number):
            match operator:
                case '*':
                    foms = first.foms
                    weights = [w * second for w in first.weights]
                case '/':
                    foms = first.foms
                    weights = [w / second for w in first.weights]
                case _:
                    raise NotImplementedError
        else:
            assert isinstance(first, Number) and isinstance(second, SuperFoM)
            match operator:
                case '*':
                    foms = second.foms
                    weights = [first * w for w in second.weights]
                case '/':
                    foms = second.foms
                    weights = [first / w for w in second.weights]
                case _:
                    raise NotImplementedError
        return SuperFoM(foms, weights)

    def __add__(self, second: Any) -> SuperFoM:
        """Return self + second."""
        return SuperFoM._math_helper(self, second, '+')

    def __radd__(self, first: Any) -> SuperFoM:
        """Return first + self."""
        return SuperFoM._math_helper(first, self, '+')

    def __iadd__(self, second: Any) -> SuperFoM:
        """Implement self += second."""
        new_fom = SuperFoM._math_helper(self, second, '+')
        self.foms = new_fom.foms
        self.weights = new_fom.weights
        # Have to create a new SuperFom so it doesn't return a FoM2
        return SuperFoM(self.foms, self.weights)

    def __sub__(self, second: Any) -> SuperFoM:
        """Return self - second."""
        return SuperFoM._math_helper(self, second, '-')

    def __rsub__(self, first: Any) -> SuperFoM:
        """Return first - self."""
        return SuperFoM._math_helper(first, self, '-')

    def __isub__(self, second: Any) -> SuperFoM:
        """Implement self -= second."""
        new_fom = SuperFoM._math_helper(self, second, '-')
        self.foms = new_fom.foms
        self.weights = new_fom.weights
        # Have to create a new SuperFom so it doesn't return a FoM2
        return SuperFoM(self.foms, self.weights)

    def __mul__(self, second: SuperFoM | Number) -> SuperFoM:
        """Return self * second."""
        return SuperFoM._math_helper(self, second, '*')

    def __rmul__(self, first: Number) -> SuperFoM:
        """Return first * self."""
        return SuperFoM._math_helper(first, self, '*')

    def __imul__(self, second: Number) -> SuperFoM:
        """Implement self *= second."""
        new_fom = SuperFoM._math_helper(self, second, '*')
        self.foms = new_fom.foms
        self.weights = new_fom.weights
        # Have to create a new SuperFom so it doesn't return a FoM2
        return SuperFoM(self.foms, self.weights)

    def __truediv__(self, second: Number) -> SuperFoM:
        """Return self / second."""
        return SuperFoM(self.foms, [w / second for w in self.weights])

    def __rtruediv__(self, first: Number) -> SuperFoM:
        """Return first / self."""
        return SuperFoM(self.foms, [first / w for w in self.weights])

    def __itruediv__(self, second: Number) -> SuperFoM:
        """Implement self /= second."""
        # FoM list doesn't need to change. Just change weights.
        self.weights = [w / second for w in self.weights]
        return SuperFoM(self.foms, self.weights)


class FoM2(SuperFoM):
    """Version 2 of FoM.

    Attributes:
        polarization (str): Polarization to use, can be "TE", "TM", or "TE+TM".
        fwd_srcs (list[Source]): The sources needed for computing the FoM; used to
            create the forward simulation.
        adj_srcs (list[Source]): The sources needed for computing the adjoint; used to
            create the adjoint simulation
        fom_monitors (list[Monitor]): The monitors needed for computing the FoM
        grad_monitors (list[Monitor]): The monitors needed for computing the gradient
        fom_func (Callable[..., npt.ArrayLike]): The function to compute the FoM.
        grad_func (Callable[..., npt.ArrayLike]): The function to compute the gradient.
        pos_max_freqs (list[int]): List of indices specifying which frequency bands are
            being maximized in the optimization.
        neg_min_freqs (list[int]): List of indices specifying which frequency bands are
            being minimized in the optimization.
    """

    def __init__(
        self,
        polarization: str,
        fwd_srcs: list[Source],
        adj_srcs: list[Source],
        fom_monitors: list[Monitor],
        grad_monitors: list[Monitor],
        fom_func: Callable[..., npt.NDArray],
        grad_func: Callable[..., npt.NDArray],
        pos_max_freqs: list[int],
        neg_min_freqs: list[int],
    ) -> None:
        """Initialize a FoM2 object."""
        super().__init__([(self,)], [1.0])
        self.fwd_srcs = fwd_srcs
        self.adj_srcs = adj_srcs
        self.fom_monitors = fom_monitors
        self.grad_monitors = grad_monitors
        self.fom_func = fom_func
        self.grad_func = grad_func
        if polarization not in POLARIZATIONS:
            raise ValueError(
                f'Polarization must be one of {POLARIZATIONS}; got {polarization}'
            )
        self.polarization = polarization
        self.pos_max_freqs = pos_max_freqs
        self.neg_min_reqs = neg_min_freqs

    def __eq__(self, other: Any) -> bool:
        """Test equality."""
        if isinstance(other, FoM2):
            return (
                self.polarization == other.polarization
                and self.fwd_srcs == other.fwd_srcs
                and self.adj_srcs == other.adj_srcs
                and self.fom_monitors == other.fom_monitors
                and self.grad_monitors == other.grad_monitors
                and self.fom_func == other.fom_func
                and self.grad_func == other.grad_func
                and self.pos_max_freqs == other.pos_max_freqs
                and self.neg_min_reqs == other.neg_min_reqs
            )
        return super().__eq__(other)

    def __copy__(self) -> FoM2:
        """Return a copy of this FoM."""
        return FoM2(
            self.polarization,
            self.fwd_srcs,
            self.adj_srcs,
            self.fom_monitors,
            self.grad_monitors,
            self.fom_func,
            self.grad_func,
            self.pos_max_freqs,
            self.neg_min_reqs,
        )

    def compute_fom(self, *args, **kwargs) -> npt.NDArray:
        """Compute the figure of merit."""
        total_fom = self.fom_func(*args, **kwargs)
        return self._subtract_neg(total_fom)

    def compute_grad(self, *args, **kwargs) -> npt.NDArray:
        """Compute the gradient of the figure of merit."""
        total_grad = self.grad_func(*args, **kwargs)
        return self._subtract_neg(total_grad)

    def _subtract_neg(self, array: npt.NDArray) -> npt.NDArray:
        """Subtract the restricted indices from the positive ones."""
        if len(self.pos_max_freqs) == 0:
            return 0 - array
        if len(self.neg_min_reqs) == 0:
            return array
        return array[..., self.pos_max_freqs] - array[..., self.neg_min_reqs]

    def create_forward_sim(self, base_sim: LumericalSimulation) -> LumericalSimulation:
        """Create a simulation with only the forward sources enabled."""
        return base_sim.with_enabled(self.fwd_srcs)

    def create_adjoint_sim(self, base_sim: LumericalSimulation) -> LumericalSimulation:
        """Create a simulation with only the adjoint sources enabled."""
        return base_sim.with_enabled(self.adj_srcs)


def unique_fwd_sim_map(foms: list[FoM2]) -> dict[frozenset[Source], list[FoM2]]:
    """Creates a map of all the unique forward sims and their corresponding FoMs."""
    sim_map: dict[frozenset[Source], list[FoM2]] = defaultdict(list)
    for fom in foms:
        fwd_srcs = frozenset(fom.fwd_srcs)
        sim_map[fwd_srcs].append(fom)
    return sim_map


def unique_adj_sim_map(foms: list[FoM2]) -> dict[frozenset[Source], list[FoM2]]:
    """Creates a map of all the unique adjoint sims and their corresponding FoMs."""
    sim_map: dict[frozenset[Source], list[FoM2]] = defaultdict(list)
    for fom in foms:
        adj_srcs = frozenset(fom.adj_srcs)
        sim_map[adj_srcs].append(fom)
    return sim_map

# TODO: Make this work for getting the lists of sources in sims
FOM_ONE = FoM2(
    'TE+TM',
    [],
    [],
    [],
    [],
    lambda _: np.ones(1),
    lambda _: np.ones(1),
    [1],
    []
)


# TODO: Strip down functionality in FoM.
# Currently, it's responsible for more than it should be in reality (storing frequency,
# polarization, and opt_ids maybe?)
# TODO: Create simulations from FoM's automatically
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
        freq: Sequence[Number],  # freq actually refers to the WHOLE lambda vector
        opt_ids: Sequence[int] | None = None,
        name: str = '',
    ) -> None:
        """Initialize an FoM."""
        self.fom_monitors = tuple(fom_monitors)
        self.grad_monitors = tuple(grad_monitors)
        self.fom_func = fom_func
        self.gradient_func = gradient_func
        self.polarization = polarization
        self.freq = freq
        self.opt_ids = tuple(range(len(freq))) if opt_ids is None else tuple(opt_ids)
        if name == '':
            self.name = f'fom_{FoM._COUNTER}'
        else:
            self.name = name
        FoM._COUNTER += 1

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, FoM):
            return (
                self.fom_monitors == __value.fom_monitors
                and self.grad_monitors == __value.grad_monitors
                and self.fom_func.__name__ == __value.fom_func.__name__
                and self.gradient_func.__name__ == __value.gradient_func.__name__
                and self.polarization == __value.polarization
                and self.freq == __value.freq
                and self.opt_ids == __value.opt_ids
            )
        return super().__eq__(__value)

    def compute_fom(self, *args, **kwargs) -> npt.NDArray:
        """Compute FoM."""
        return self.fom_func(*args, **kwargs)

    def compute_gradient(
        self, *args, **kwargs
    ) -> npt.NDArray:  # ! 20240227 Ian - Renamed this.
        """Compute gradient of FoM."""
        return self.gradient_func(*args, **kwargs)

    def as_dict(self) -> dict:
        """Return a dictionary representation of this FoM."""
        data: dict[str, Any] = {}
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
    ) -> FoM:
        """Create a figure of merit from a dictionary and list of simulations."""
        data = copy(og_data)
        fom_cls: type[FoM] = getattr(sys.modules[__name__], data.pop('type'))
        if 'weight' in data:
            del data['weight']
        data['name'] = name
        # if 'fom_func' in data:
        # if 'gradient_func' in data:

        data['fom_monitors'] = [
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
                raise NotImplementedError(
                    f'Unrecognized operation "{operator}";'
                    r' choose one of \{+, -, /, *\} '
                )

        if isinstance(second, Number):
            og_fom_func = first.fom_func
            og_grad_func = first.gradient_func
            return FoM(
                first.fom_monitors,
                first.grad_monitors,
                # lambda *args, **kwargs: func(og_fom_func(*args, **kwargs), second),
                # lambda *args, **kwargs: func(og_grad_func(*args, **kwargs), second),
                lambda *args, **kwargs: tuple(
                    func(x, second) for x in og_fom_func(*args, **kwargs)
                ),
                lambda *args, **kwargs: tuple(
                    func(x, second) for x in og_grad_func(*args, **kwargs)
                ),
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
                lambda *args, **kwargs: tuple(
                    func(first, x) for x in og_fom_func(*args, **kwargs)
                ),
                lambda *args, **kwargs: tuple(
                    func(first, x) for x in og_grad_func(*args, **kwargs)
                ),
                second.polarization,
                second.freq,
                second.opt_ids,
            )

        assert isinstance(first, FoM)
        assert isinstance(second, FoM)

        # NOTE: Polarization might be unnecessary as a check
        if (
            first.polarization != second.polarization
            or first.opt_ids != second.opt_ids
            or first.freq != second.freq
        ):
            raise TypeError(
                f"unsupported operand type(s) for {operator}: 'FoM' and 'FoM' when "
                'polarization, opt_ids, and/or freq is not equal'
            )

        # og_fom_func_1 = first.fom_func
        # og_grad_func_1 = first.gradient_func
        # og_fom_func_2 = second.fom_func
        # og_grad_func_2 = second.gradient_func

        def new_compute(*args, **kwargs):
            first_fom = first.compute_fom(*args, **kwargs)
            second_fom = second.compute_fom(*args, **kwargs)
            if len(first_fom) == len(second_fom) > 1:
                res = tuple(starmap(func, zip(first_fom, second_fom, strict=True)))
            elif len(first_fom) > len(second_fom):
                res = tuple(func(x, second_fom) for x in first_fom)
            elif len(first_fom) < len(second_fom):
                res = tuple(func(first_fom, x) for x in second_fom)
            else:
                res = tuple(func(first_fom, second_fom))
            return res

        def new_gradient(*args, **kwargs):
            first_grad = first.compute_gradient(*args, **kwargs)
            second_grad = second.compute_gradient(*args, **kwargs)
            if len(first_grad) == len(second_grad) > 1:
                res = tuple(starmap(func, zip(first_grad, second_grad, strict=True)))
            elif len(first_grad) > len(second_grad):
                res = tuple(func(x, second_grad) for x in first_grad)
            elif len(first_grad) < len(second_grad):
                res = tuple(func(first_grad, x) for x in second_grad)
            else:
                res = tuple(func(first_grad, second_grad))
            return res

        return FoM(
            first.grad_monitors + second.grad_monitors,
            first.fom_monitors + second.fom_monitors,
            new_compute,
            new_gradient,
            first.polarization,
            first.freq,
            opt_ids=first.opt_ids,
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

        def zero_func(*args, **kwargs):
            return np.zeros(1)

        return FoM(
            [],
            [],
            zero_func,
            zero_func,
            fom.polarization,
            fom.freq,
            fom.opt_ids,
        )

    def __copy__(self) -> FoM:
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


""" TODO: add the following
self.freq_index_negative_opt:
    Specify and array of frequency indices that should be optimized with a negative
    gradient. This is useful for making sure light does not focus to a point,
    for example.
self.fom = np.array([]):
    The more convenient way to look at fom. For example, power transmission through a
    monitor even if you're optimizing for a point source.
self.restricted_fom = np.array([])
    FOM that is being restricted. For instance, frequencies that should not be focused.
self.true_fom = np.array([])
    The true FoM being used to define the adjoint source.
self.gradient = np.array([])
self.restricted_gradient = np.array([])

self.tempfile_fwd_name = ''
self.tempfile_adj_name = ''

self.design_fwd_fields = None
self.design_adj_fields = None

Boolean array specifying which frequencies are active. This is a bit confusing. Almost
deprecated really. Enabled means of the frequencies being optimized, which are enabled.
Useful in rare circumstances where some things need to be fully disable to help catch up
self.enabled = np.ones((len(self.freq_index_opt)))

Adding this once I started optimizing for functions we DONT want
(i.e. restricted_gradient). This is of the freq_index_opt_restricted values,
which do we want to optimize?
self.enabled_restricted = np.ones((len(self.freq_index_restricted_opt)))
"""


class BayerFilterFoM(FoM):
    """FoM implementing the particular figure of merit for the SonyBayerFilter."""

    def __init__(
        self,
        fom_monitors: Sequence[Monitor],
        grad_monitors: Sequence[Monitor],
        polarization: str,
        freq: Sequence[Number],
        opt_ids: Sequence[int] | None = None,
        name: str = '',
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
        # Here we need to figure out which FoM belongs to which file
        # FoM for transmission monitor
        total_tfom = np.zeros(self.fom_monitors[0].tshape)[..., self.opt_ids]
        # FoM for focal monitor - take [1:] because intensity sums over first axis
        total_ffom = np.zeros(self.fom_monitors[0].fshape[1:])[..., self.opt_ids]

        # Source weight calculation.
        source_weight = np.zeros(self.fom_monitors[0].fshape, dtype=np.complex128)
        for monitor in self.fom_monitors:
            transmission = monitor.trans_mag
            vipdopt.logger.info(f'Accessing monitor {monitor.monitor_name}')
            # print(transmission.shape)
            # print(self.opt_ids)
            # print(total_tfom.shape)
            # print(transmission[..., self.opt_ids].shape)
            # print(transmission[..., self.opt_ids])
            total_tfom += transmission[..., self.opt_ids]

            efield = monitor.e
            total_ffom += np.sum(np.square(np.abs(efield[..., self.opt_ids])), axis=0)

            source_weight += np.expand_dims(
                np.conj(efield[:, 0, 0, 0, :]), axis=(1, 2, 3)
            )
            # We'll apply opt_ids slice when gradient is fully calculated.
            # source_weight += np.conj(efield[:,0,0,0, self.opt_ids])

        # Recall that E_adj = source_weight * what we call E_adj i.e. the Green's
        # function[design_efield from adj_src simulation]. Reshape source weight (nλ)
        # to (1, 1, 1, nλ) so it can be multiplied with (E_fwd * E_adj) https://stackoverflow.com/a/30032182
        self.source_weight = (
            source_weight  # np.expand_dims(source_weight, axis=(1,2,3))
        )
        # TODO: Ian - I don't want to mess with the return types for _bayer_fom so I assigned source_weight to the fom object itself

        # Conjugate of E_{old}(x_0) -field at the adjoint source of interest, with
        # direction along the polarization. This is going to be the amplitude of the
        # dipole-adjoint source driven at the focal plane. Reminder that this is only
        # applicable for dipole-based adjoint sources

        # x-polarized if phi = 0, y-polarized if phi = 90.
        # pol_xy_idx = 0 if adj_src.src_dict['phi'] == 0 else 1
        # TODO: REDO - direction of source_weight vector potential error.

        # self.source_weight = np.squeeze(
        #     np.conj(
        #         focal_data[pol_xy_idx, 0, 0, 0, :]  # shape: (3, nx, ny, nz, nλ)
        # 		get_focal_data[adj_src_idx][
        #             xy_idx,
        #             0,
        #             0,
        #             0,
        #             spectral_indices[0]:spectral_indices[1]:1,
        #         ]
        #     )
        # )
        # self.source_weight += np.squeeze( np.conj( focal_data[:,0,0,0,:] ) )

        return total_tfom, total_ffom

    def _bayer_gradient(self):
        """Compute the gradient of the bayer filter figure of merit."""
        e_fwd = self.design_fwd_fields
        e_adj = self.grad_monitors[1].e

        # #! DEBUG: Check orthogonality and direction of E-fields in the design monitor
        vipdopt.logger.info(
            f'Forward design fields have average absolute xyz-components: '
            f'{np.mean(np.abs(e_fwd[0]))}, {np.mean(np.abs(e_fwd[1]))}, '
            f'{np.mean(np.abs(e_fwd[2]))}.'
        )
        vipdopt.logger.info(
            f'Adjoint design fields have average absolute xyz-components: '
            f'{np.mean(np.abs(e_adj[0]))}, {np.mean(np.abs(e_adj[1]))}, '
            f'{np.mean(np.abs(e_adj[2]))}.'
        )
        vipdopt.logger.info(
            f'Source weight has average absolute xyz-components: '
            f'{np.mean(np.abs(self.source_weight[0]))}, '
            f'{np.mean(np.abs(self.source_weight[1]))}, '
            f'{np.mean(np.abs(self.source_weight[2]))}.'
        )

        # df_dev = np.real(np.sum(e_fwd * e_adj, axis=0))
        e_adj = e_adj * self.source_weight
        df_dev = 1 * (e_fwd[0] * e_adj[0] + e_fwd[1] * e_adj[1] + e_fwd[2] * e_adj[2])
        # Taking real part comes when multiplying by Δε0 i.e. change in permittivity.

        vipdopt.logger.info('Computing Gradient')

        self.gradient = np.zeros(df_dev.shape, dtype=np.complex128)
        # self.restricted_gradient = np.zeros(df_dev.shape, dtype=np.complex128)

        self.gradient[..., self.opt_ids] = df_dev[..., self.opt_ids]  # * self.enabled
        # self.restricted_gradient[..., self.freq_index_restricted_opt] = \
        #     df_dev[..., self.freq_index_restricted_opt] * self.enabled_restricted

        # self.gradient = df_dev[..., pos_gradient_indices] * self.enabled
        # self.restricted_gradient = df_dev[..., neg_gradient_indices] * \
        #       self.enabled_restricted

        # return df_dev

        return df_dev[..., self.opt_ids]


class UniformFoM(FoM):
    """A figure of merit for a device with uniform density."""

    def __init__(
        self,
        fom_monitors: Sequence[Monitor],
        grad_monitors: Sequence[Monitor],
        polarization: str,
        freq: Sequence[Number],
        opt_ids: Sequence[int] | None = None,
        name: str = '',
        constant: float = 0.5,
    ) -> None:
        """Initialize a UniformFoM."""
        self.constant = constant
        super().__init__(
            fom_monitors,
            grad_monitors,
            self._uniform_fom,
            self._uniform_gradient,
            polarization,
            freq,
            opt_ids,
            name,
        )

    def _uniform_fom(self, variables: npt.NDArray):
        return 1 - np.abs(variables - self.constant)

    def _uniform_gradient(self, variables: npt.NDArray):
        return np.sign(variables - self.constant)
