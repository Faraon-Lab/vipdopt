"""Code for Figures of Merit (FoMs)."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from copy import copy
from functools import reduce
from itertools import product
from typing import Any, Concatenate

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt.simulation import LumericalSimulation, Monitor, Source
from vipdopt.simulation.monitor import Power, Profile
from vipdopt.simulation.source import DipoleSource, GaussianSource
from vipdopt.utils import (
    Number,
    P,
    flatten,
    import_lumapi,
    setup_logger,
    starmap_with_kwargs,
)

POLARIZATIONS = ['TE', 'TM', 'TE+TM']


class SuperFoM:
    """Representation of a weighted sum of FoMs that take the same arguments.

    A SuperFoM stores a list of tuples of FoMs. When computing the overall FoM, the
    SuperFoM returns the weighted sum of all its parts. Tuples containing multiple
    FoMs represent the product of each element. For example, if you have FoMs (X, Y) and
    (Z,) with weights k and j respectively, the output would be kXY + jZ.

    Attributes:
        foms (list[tuple[FoM, ...]]): The (groups of) FoMs contained in the overall FoM
        weights (list[float]): The weights to apply to each FoM
    """

    def __init__(
        self, foms: Sequence[Iterable[FoM]], weights: Sequence[float] = (1.0,)
    ) -> None:
        """Initialize a SuperFoM."""
        self.foms: list[tuple[FoM, ...]] = [tuple(f) for f in foms]
        self.weights: list[float] = list(weights)
        self.performance_weights = np.ones(len(self.foms))

    def __copy__(self) -> SuperFoM:
        """Create a copy of this FoM."""
        return SuperFoM(self.foms, self.weights)

    def __eq__(self, other: Any) -> bool:
        """Test equality."""
        if isinstance(other, SuperFoM):
            return self.foms == other.foms and self.weights == other.weights
        return super().__eq__(other)

    def reset_monitors(self):
        """Reset all of the monitors used to calculate the FoM."""
        map(FoM.reset_monitors, flatten(self.foms))

    def performance_weighting(self, fom_values: npt.NDArray):
        """Recompute the weights based on the performance of the optimization.

        All gradients are combined with a weighted average in Eq.(3), with weights
        chosen according to Eq.(2) such that all figures of merit seek the same
        efficiency. In these equations, FoM represents the current value of a figure of
        merit, N is the total number of figures of merit, and wi represents the weight
        applied to its respective merit function's gradient. The maximum operator is
        used to ensure the weights are never negative, thus ignoring the gradient of
        high-performing figures of merit rather than forcing the figure of merit to
        decrease. The 2/N factor is used to ensure all weights conveniently sum to 1
        unless some weights were negative before the maximum operation. Although the
        maximum operator is non-differentiable, this function is used only to apply the
        gradient rather than to compute it. Therefore, it does not affect the
        applicability of the adjoint method. Taken from: https://doi.org/10.1038/s41598-021-88785-5

        Arguments:
            fom_values (npt.NDArray): The values of the computed FoMs to determine the
                new weights from. Should have shape 1 x N where N is the number of
                FoMs.
        """
        weights = (2.0 / len(fom_values)) - fom_values**2 / np.sum(fom_values**2)

        # Zero-shift and renormalize
        if np.min(weights) < 0:
            weights -= np.min(weights)
            weights /= np.sum(weights)

        self.performance_weights = weights

    def compute_fom(self, *args, **kwargs) -> npt.NDArray:
        """Compute the weighted sum of the FoMs."""
        fom_results = np.array([
            SuperFoM._compute_prod(
                FoM.compute_fom,
                fom_tup,
                *args,
                **kwargs,
            )
            for fom_tup in self.foms
        ])
        self.performance_weighting(fom_results)
        # fom_results = np.dot(fom_results, spectral_weights).dot(performance_weights)
        return np.einsum('i,i...->...', self.weights, fom_results)

    @staticmethod
    def _compute_prod(
        function: Callable, foms: tuple[FoM, ...], *args, **kwargs
    ) -> npt.NDArray:
        """Compute the product of all FoMs contained inside a group."""
        factors = np.array(
            list(
                starmap_with_kwargs(
                    function, ((fom, *args) for fom in foms), (kwargs for _ in foms)
                )
            )
        )
        return np.prod(factors, axis=0)

    @staticmethod
    def _prod_rule(foms: tuple[FoM, ...], *args, **kwargs) -> npt.NDArray:
        """Apply the product rule for differentiation."""
        if len(foms) == 1:
            return FoM.compute_grad(foms[0], *args, **kwargs)
        # Otherwise we need to use the product rule
        fom_vals = np.array(
            list(
                starmap_with_kwargs(
                    FoM.compute_fom,
                    ((fom, *args) for fom in foms),
                    ({'sum_values': False, **kwargs} for _ in foms),
                )
            )
        )
        grad_vals = np.array(
            list(
                starmap_with_kwargs(
                    FoM.compute_grad,
                    ((fom, *args) for fom in foms),
                    (kwargs for _ in foms),
                )
            )
        )
        term2 = np.sum(
            np.divide(
                grad_vals,
                fom_vals.reshape(grad_vals.shape),
                out=np.zeros(grad_vals.shape),
                where=fom_vals != 0,  # Return zeros where division by zero occur
                dtype=float,
            ),
            axis=0,
        )
        return np.prod(fom_vals, axis=0) * term2

    def compute_grad(
        self, *args, apply_performance_weights=False, **kwargs
    ) -> npt.NDArray:
        """Compute the weighted sum of the gradients."""
        grad_results = np.array([
            SuperFoM._prod_rule(
                fom_tup,
                *args,
                **kwargs,
            )
            for fom_tup in self.foms
        ])
        # grad_results = np.dot(grad_results, spectral_weights).dot(performance_weights)
        if apply_performance_weights:
            return np.einsum('i,i...->...', self.performance_weights, grad_results)
        return np.einsum('i,i...->...', self.weights, grad_results)

    def create_forward_sim(
        self, base_sim: LumericalSimulation
    ) -> list[LumericalSimulation]:
        """Create all unique forward simulations needed to compute this FoM."""
        fwd_sim_map = unique_fwd_sim_map(flatten(self.foms))
        sims = [
            base_sim.with_enabled(
                srcs,
                base_sim.info['name']
                + '_fwd_'
                + '_'.join(src.name for src in sorted(srcs)),
            )
            for srcs in fwd_sim_map
        ]
        for i, foms in enumerate(fwd_sim_map.values()):
            for fom in foms:
                fom.link_forward_sim(sims[i])
        return sims

    def create_adjoint_sim(
        self, base_sim: LumericalSimulation
    ) -> list[LumericalSimulation]:
        """Create all unique adjoint simulations needed to compute this FoM."""
        adj_sim_map = unique_adj_sim_map(flatten(self.foms))
        sims = [
            base_sim.with_enabled(
                srcs,
                base_sim.info['name']
                + '_adj_'
                + '_'.join(src.name for src in sorted(srcs)),
            )
            for srcs in adj_sim_map
        ]
        for i, foms in enumerate(adj_sim_map.values()):
            for fom in foms:
                fom.link_adjoint_sim(sims[i])
        return sims

    @staticmethod
    def _math_helper(
        first: SuperFoM | Number, second: SuperFoM | Number, operator: str
    ) -> SuperFoM:
        """Helper function for computing arithmetic functions.

        The supported arithmetic functions include:
        * addition / subtraction of two SuperFoMs
        * multiplication of two SuperFoMs
        * scalar multiplication
        * division by a scalar
        """
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
                    foms = [
                        tuple(flatten(tup)) for tup in product(first.foms, second.foms)
                    ]
                    weights = [
                        reduce(lambda x, y: x * y, tup)
                        for tup in product(first.weights, second.weights)
                    ]
                case _:
                    return NotImplemented
        elif isinstance(first, SuperFoM) and isinstance(second, Number):
            match operator:
                case '+':
                    foms = [tuple([second])] + first.foms
                case '*':
                    foms = first.foms
                    weights = [w * second for w in first.weights]
                case '/':
                    foms = first.foms
                    weights = [w / second for w in first.weights]
                case _:
                    return NotImplemented
        elif isinstance(first, Number) and isinstance(second, SuperFoM):
            match operator:
                case '+':
                    foms = [tuple([first])] + second.foms
                    weights = [1.0] + second.weights
                case '*':
                    foms = second.foms
                    weights = [first * w for w in second.weights]
                case _:
                    raise NotImplementedError
        else:
            return NotImplemented
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

    def __mul__(self, second: Any) -> SuperFoM:
        """Return self * second."""
        return SuperFoM._math_helper(self, second, '*')

    def __rmul__(self, first: Any) -> SuperFoM:
        """Return first * self."""
        return SuperFoM._math_helper(first, self, '*')

    def __imul__(self, second: Any) -> SuperFoM:
        """Implement self *= second."""
        new_fom = SuperFoM._math_helper(self, second, '*')
        self.foms = new_fom.foms
        self.weights = new_fom.weights
        # Have to create a new SuperFom so it doesn't return a FoM2
        return SuperFoM(self.foms, self.weights)

    def __truediv__(self, second: Any) -> SuperFoM:
        """Return self / second."""
        return SuperFoM._math_helper(self, second, '/')

    def __itruediv__(self, second: Any) -> SuperFoM:
        """Implement self /= second."""
        new_fom = SuperFoM._math_helper(self, second, '/')
        self.foms = new_fom.foms
        self.weights = new_fom.weights
        # Have to create a new SuperFom so it doesn't return a FoM2
        return SuperFoM(self.foms, self.weights)

    def as_dict(self) -> dict:
        """Return a dictionary representation of this FoM."""
        data: dict[str, Any] = {}
        data['foms'] = self.foms
        data['weights'] = self.weights

        return data

    @staticmethod
    def from_dict(data: dict) -> SuperFoM:
        """Create a SuperFom from a dictionary representation."""
        return SuperFoM(data['foms'], data['weights'])


class FoM(SuperFoM):
    """Generic class for computing a figure of merit (FoM).

    Supports arithmetic operations on class instances, either with each other,
    or with scalar values.

    Attributes:
        polarization (str): Polarization to use, can be "TE", "TM", or "TE+TM".
        fwd_srcs (list[Source]): The sources needed for computing the FoM; used to
            create the forward simulation.
        adj_srcs (list[Source]): The sources needed for computing the adjoint; used to
            create the adjoint simulation
        fom_monitors (list[Monitor]): The monitors to track in the forward simulation.
        adj_monitors (list[Monitor]): The monitors to track in the adjoint simulation.
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
        fwd_monitors: list[Monitor],
        adj_monitors: list[Monitor],
        fom_func: Callable[Concatenate[FoM, P], npt.NDArray],
        grad_func: Callable[Concatenate[FoM, P], npt.NDArray],
        pos_max_freqs: Sequence[int],
        neg_min_freqs: Sequence[int],
        spectral_weights: npt.NDArray = np.array(1),
    ) -> None:
        """Initialize a FoM object."""
        super().__init__([(self,)], [1.0])
        self.fwd_srcs = fwd_srcs
        self.adj_srcs = adj_srcs
        self.fwd_monitors = fwd_monitors
        self.adj_monitors = adj_monitors
        self.fom_func = fom_func
        self.grad_func = grad_func
        if polarization not in POLARIZATIONS:
            raise ValueError(
                f'Polarization must be one of {POLARIZATIONS}; got {polarization}'
            )
        self.polarization = polarization
        self.pos_max_freqs = list(pos_max_freqs)
        self.neg_min_freqs = list(neg_min_freqs)
        self.spectral_weights = spectral_weights

    def __eq__(self, other: Any) -> bool:
        """Test equality."""
        if isinstance(other, FoM):
            return (
                self.polarization == other.polarization
                and self.fwd_srcs == other.fwd_srcs
                and self.adj_srcs == other.adj_srcs
                and self.fwd_monitors == other.fwd_monitors
                and self.adj_monitors == other.adj_monitors
                and self.fom_func == other.fom_func
                and self.grad_func == other.grad_func
                and self.pos_max_freqs == other.pos_max_freqs
                and self.neg_min_freqs == other.neg_min_freqs
                and self.spectral_weights == other.spectral_weights
            )
        return super().__eq__(other)

    def __copy__(self) -> FoM:
        """Return a copy of this FoM."""
        return FoM(
            self.polarization,
            self.fwd_srcs,
            self.adj_srcs,
            self.fwd_monitors,
            self.adj_monitors,
            self.fom_func,
            self.grad_func,
            self.pos_max_freqs,
            self.neg_min_freqs,
            self.spectral_weights,
        )

    def reset_monitors(self):
        """Reset all of the monitors used to calculate the FoM."""
        for mon in self.fwd_monitors:
            mon.reset()
        for mon in self.adj_monitors:
            mon.reset()

    def compute_fom(self, *args, sum_values: bool = True, **kwargs) -> npt.NDArray:
        """Compute the figure of merit."""
        total_fom = self.fom_func(*args, **kwargs)
        self.reset_monitors()
        # return self._subtract_neg(total_fom)
        f = np.dot(total_fom, self.spectral_weights)
        if sum_values:
            return f.sum()
        return f

    def compute_grad(self, *args, **kwargs) -> npt.NDArray:
        """Compute the gradient of the figure of merit."""
        total_grad = self.grad_func(*args, **kwargs)
        self.reset_monitors()
        # return self._subtract_neg(total_grad)
        return np.dot(total_grad, self.spectral_weights)

    def _subtract_neg(self, array: npt.NDArray) -> npt.NDArray:
        """Subtract the restricted indices from the positive ones."""
        if len(self.pos_max_freqs) == 0:
            return 0 - array
        if len(self.neg_min_freqs) == 0:
            return array
        return array[..., self.pos_max_freqs] - array[..., self.neg_min_freqs]

    def link_forward_sim(self, sim: LumericalSimulation):
        """Link this FoM's fwd_monitors to a provided simulation."""
        self.fwd_monitors = [sim.objects[m.name] for m in self.fwd_monitors]

    def create_forward_sim(
        self, base_sim: LumericalSimulation
    ) -> list[LumericalSimulation]:
        """Create a simulation with only the forward sources enabled."""
        new_name = (
            base_sim.info['name']
            + '_fwd_'
            + '_'.join(src.name for src in self.fwd_srcs)
        )
        fwd_sim = base_sim.with_enabled(self.fwd_srcs, new_name)
        self.link_forward_sim(fwd_sim)
        return [fwd_sim]

    def link_adjoint_sim(self, sim: LumericalSimulation):
        """Link this FoM's adj_monitors to a provided simulation."""
        self.adj_monitors = [sim.objects[m.name] for m in self.adj_monitors]

    def create_adjoint_sim(
        self, base_sim: LumericalSimulation
    ) -> list[LumericalSimulation]:
        """Create a simulation with only the adjoint sources enabled."""
        new_name = (
            base_sim.info['name']
            + '_fwd_'
            + '_'.join(src.name for src in self.fwd_srcs)
        )
        adj_sim = base_sim.with_enabled(self.adj_srcs, new_name)
        self.link_adjoint_sim(adj_sim)
        return [adj_sim]

    def as_dict(self) -> dict:
        """Return a dictionary representation of this FoM."""
        data: dict = {}
        data['type'] = type(self).__name__
        data['polarization'] = self.polarization
        data['fwd_srcs'] = self.fwd_srcs
        data['adj_srcs'] = self.adj_srcs
        data['fwd_monitors'] = self.fwd_monitors
        data['adj_monitors'] = self.adj_monitors
        if data['type'] == 'FoM':  # Generic FoM needs to copy functions
            data['fom_func'] = self.fom_func
            data['grad_func'] = self.grad_func
        data['pos_max_freqs'] = self.pos_max_freqs
        data['neg_min_freqs'] = self.neg_min_freqs

        return data

    @staticmethod
    def from_dict(input_dict: dict) -> FoM:
        """Create a FoM from a dictionary representation."""
        data = copy(input_dict)
        fom_cls: type[FoM] = getattr(sys.modules[__name__], data.pop('type'))
        return fom_cls(**data)


def unique_fwd_sim_map(foms: Iterable[FoM]) -> dict[frozenset[Source], list[FoM]]:
    """Creates a map of all the unique forward sims and their corresponding FoMs."""
    sim_map: dict[frozenset[Source], list[FoM]] = defaultdict(list)
    for fom in foms:
        fwd_srcs = frozenset(fom.fwd_srcs)
        sim_map[fwd_srcs].append(fom)
    return sim_map


def unique_adj_sim_map(foms: Iterable[FoM]) -> dict[frozenset[Source], list[FoM]]:
    """Creates a map of all the unique adjoint sims and their corresponding FoMs."""
    sim_map: dict[frozenset[Source], list[FoM]] = defaultdict(list)
    for fom in foms:
        adj_srcs = frozenset(fom.adj_srcs)
        sim_map[adj_srcs].append(fom)
    return sim_map


class BayerFilterFoM(FoM):
    """FoM implementing the particular figure of merit for the SonyBayerFilter.

    Must have the following monitor configuration:
        fwd_monitors: [focal_monitor, transmission_monitor, design_efield]
        adj_monitors: [design_efield]

    """

    def __init__(
        self,
        polarization: str,
        fwd_srcs: list[Source],
        adj_srcs: list[Source],
        fwd_monitors: list[Monitor],
        adj_monitors: list[Monitor],
        pos_max_freqs: list[int],
        neg_min_freqs: list[int],
        spectral_weights: npt.NDArray = np.array(1),
    ) -> None:
        """Initialize a BayerFilterFoM."""
        super().__init__(
            polarization,
            fwd_srcs,
            adj_srcs,
            fwd_monitors,
            adj_monitors,
            self._bayer_fom,
            self._bayer_gradient,
            pos_max_freqs,
            neg_min_freqs,
            spectral_weights,
        )

    def _bayer_fom(self):
        """Compute bayer filter figure of merit."""
        # Here we need to figure out which FoM belongs to which file

        # for mon in self.fwd_monitors:
        #     vipdopt.logger.debug(vars(mon))

        # FoM for transmission monitor
        total_tfom = np.zeros(self.fwd_monitors[1].tshape)[..., self.pos_max_freqs]
        # FoM for focal monitor - take [1:] because intensity sums over first axis
        total_ffom = np.zeros(self.fwd_monitors[0].fshape[1:])[..., self.pos_max_freqs]

        # Source weight calculation.
        source_weight = np.zeros(self.fwd_monitors[0].fshape, dtype=np.complex128)
        # for monitor in self.fwd_monitors:
        transmission = self.fwd_monitors[1].trans_mag
        # vipdopt.logger.info(f'Accessing monitor {monitor.monitor_name}')
        # print(transmission.shape)
        # print(self.opt_ids)
        # print(total_tfom.shape)
        # print(transmission[..., self.opt_ids].shape)
        # print(transmission[..., self.opt_ids])
        total_tfom += transmission[..., self.pos_max_freqs]

        efield = self.fwd_monitors[0].e
        total_ffom += np.sum(np.square(np.abs(efield[..., self.pos_max_freqs])), axis=0)

        source_weight += np.expand_dims(np.conj(efield[:, 0, 0, 0, :]), axis=(1, 2, 3))
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

        # return total_tfom, total_ffom
        return total_ffom

    def _bayer_gradient(self):
        """Compute the gradient of the bayer filter figure of merit."""
        # e_fwd = self.design_fwd_fields
        e_fwd = self.fwd_monitors[2].e
        e_adj = self.adj_monitors[0].e

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

        self.gradient[..., self.pos_max_freqs] = df_dev[
            ..., self.pos_max_freqs
        ]  # * self.enabled
        # self.restricted_gradient[..., self.freq_index_restricted_opt] = \
        #     df_dev[..., self.freq_index_restricted_opt] * self.enabled_restricted

        # self.gradient = df_dev[..., pos_gradient_indices] * self.enabled
        # self.restricted_gradient = df_dev[..., neg_gradient_indices] * \
        #       self.enabled_restricted

        # return df_dev

        return df_dev


class UniformMAEFoM(FoM):
    """A figure of merit for a uniform density using mean absolute error."""

    def __init__(
        self,
        polarization: str,
        fwd_srcs: list[Source],
        adj_srcs: list[Source],
        fwd_monitors: list[Monitor],
        adj_monitors: list[Monitor],
        pos_max_freqs: list[int],
        neg_min_freqs: list[int],
        constant: float,
        spectral_weights: npt.NDArray = np.array(1),
    ) -> None:
        """Initialize a UniformFoM."""
        super().__init__(
            polarization,
            fwd_srcs,
            adj_srcs,
            fwd_monitors,
            adj_monitors,
            self._uniform_mae_fom,
            self._uniform_mae_gradient,
            pos_max_freqs,
            neg_min_freqs,
            spectral_weights=spectral_weights,
        )
        self.constant = constant

    def _uniform_mae_fom(self, x: npt.NDArray):
        return 1 - np.abs(x - self.constant)

    def _uniform_mae_gradient(self, x: npt.NDArray):
        return np.sign(self.constant - x)


class UniformMSEFoM(FoM):
    """A figure of merit for a uniform density using mean squared error."""

    def __init__(
        self,
        polarization: str,
        fwd_srcs: list[Source],
        adj_srcs: list[Source],
        fwd_monitors: list[Monitor],
        adj_monitors: list[Monitor],
        pos_max_freqs: list[int],
        neg_min_freqs: list[int],
        constant: float,
        spectral_weights: npt.NDArray = np.array(1),
    ) -> None:
        """Initialize a UniformFoM."""
        super().__init__(
            polarization,
            fwd_srcs,
            adj_srcs,
            fwd_monitors,
            adj_monitors,
            self._uniform_mse_fom,
            self._uniform_mse_gradient,
            pos_max_freqs,
            neg_min_freqs,
            spectral_weights=spectral_weights,
        )
        self.constant = constant

    def _uniform_mse_fom(self, x: npt.NDArray):
        return 1 - np.square(x - self.constant)
    def _uniform_mse_fom(self, x: npt.NDArray):
        return 1 - np.square(x - self.constant)

    def _uniform_mse_gradient(self, x: npt.NDArray):
        return self.constant - x


def gaussian_kernel(length=5, sigma=1.0) -> npt.NDArray:
    """Creates a 2D gaussian kernel."""
    ax = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


class GaussianFoM(FoM):
    """A figure of merit for a device to match a 2d Gaussian."""

    def __init__(
        self,
        polarization: str,
        fwd_srcs: list[Source],
        adj_srcs: list[Source],
        fwd_monitors: list[Monitor],
        adj_monitors: list[Monitor],
        pos_max_freqs: list[int],
        neg_min_freqs: list[int],
        length: float,
        sigma: float,
        spectral_weights: npt.NDArray = np.array(1),
    ) -> None:
        """Initialize a UniformFoM."""
        super().__init__(
            polarization,
            fwd_srcs,
            adj_srcs,
            fwd_monitors,
            adj_monitors,
            self._gaussian_fom,
            self._gaussian_gradient,
            pos_max_freqs,
            neg_min_freqs,
            spectral_weights=spectral_weights,
        )
        self.kernel = gaussian_kernel(length, sigma)

    def _gaussian_fom(self, x: npt.NDArray):
        return 1 - np.square(x - self.kernel[..., np.newaxis])

    def _gaussian_gradient(self, x: npt.NDArray):
        return 2 * (self.kernel[..., np.newaxis] - x)


if __name__ == '__main__':
    vipdopt.logger = setup_logger('logger', 0)
    vipdopt.lumapi = import_lumapi(
        'C:\\Program Files\\Lumerical\\v221\\api\\python\\lumapi.py'
    )
    base_sim = LumericalSimulation('test_data\\sim.json')
    base_sim.set_path('test_data\\')
    srcs = base_sim.sources()
    vipdopt.logger.debug([src.name for src in srcs])
    fom = BayerFilterFoM(
        'TE',
        [GaussianSource('forward_src_x')],
        [GaussianSource('forward_src_x'), DipoleSource('adj_src_0x')],
        [
            Power('focal_monitor_0'),
            Power('transmission_monitor_0'),
            Profile('design_efield_monitor'),
        ],
        [Profile('design_efield_monitor')],
        list(range(60)),
        [],
    )
    fwd_sim = fom.create_forward_sim(base_sim)[0]
    fwd_sim.set_path('test_data\\fwd_sim.fsp')
    adj_sim = fom.create_adjoint_sim(base_sim)[0]
    adj_sim.set_path('test_data\\adj_sim.fsp')

    # fdtd = LumericalFDTD()
    # fdtd.connect(hide=True)

    # fdtd.save('test_data\\fwd_sim.fsp', fwd_sim)
    # fdtd.save('test_data\\adj_sim.fsp', adj_sim)
    # fdtd.addjob('test_data\\fwd_sim.fsp')
    # fdtd.addjob('test_data\\adj_sim.fsp')
    # fdtd.runjobs(0)

    # fdtd.reformat_monitor_data([fwd_sim, adj_sim])
    fwd_sim.link_monitors()
    adj_sim.link_monitors()

    fom_val = fom.compute_fom()
    vipdopt.logger.debug(f'FoM: {fom_val.shape}')
    grad_val = fom.compute_grad()
    vipdopt.logger.debug(f'Gradient: {grad_val.shape}')
