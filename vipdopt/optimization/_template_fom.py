
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
from vipdopt.simulation import Simulation, Monitor, Source
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


class FoM:
    
    def __init__(self, 
            fom_func: Callable[Concatenate[FoM, P], npt.NDArray],
            grad_func: Callable[Concatenate[FoM, P], npt.NDArray],
            foms: Sequence[Iterable[FoM]], 
            weights: Sequence[float] = (1.0,), 
            fwd_srcs:list[Source] = [],
            fwd_monitors:list[Monitor] = [],
            adj_srcs:list[Source] = [],
            adj_monitors:list[Monitor] = [],
            polarization:str='TE',
            pos_max_freqs: Sequence[int] = [],      # wrap this somehow into the Optimization?
            neg_min_freqs: Sequence[int] = [],      # as max/minimization should be chosen external of the FoM
            all_freqs: Sequence[float] = [],
            # spectral_weights, # wrap into weights
            # reduce_func,
            *args, **kwargs,
        ) -> None:
        """Initialize a FoM object."""
        
        self.foms: list[tuple[FoM,...]] = [tuple(f) for f in foms]
        self.fom_func = fom_func
        self.grad_func = grad_func
        self.weights: list[float] = list(weights)
        self.performance_weights = np.ones(len(self.foms))
        self.fwd_srcs = fwd_srcs
        self.fwd_monitors = fwd_monitors
        self.adj_srcs = adj_srcs
        self.adj_monitors = adj_monitors
        
        if polarization not in POLARIZATIONS:
            raise ValueError(
                f'Polarization must be one of {POLARIZATIONS}; got {polarization}'
            )
        self.polarization = polarization
        self.pos_max_freqs = list(pos_max_freqs)
        self.neg_min_freqs = list(neg_min_freqs)
        self.all_freqs = list(all_freqs)
        # self.spectral_weights = spectral_weights
        # self.reduce_func = reduce_func
        
    def __eq__(self, other: Any) -> bool:
        """Test equality."""
        if isinstance(other, FoM):
            return (
                self.foms == other.foms
                and self.weights == other.weights
                and self.polarization == other.polarization
                and self.fwd_srcs == other.fwd_srcs
                and self.adj_srcs == other.adj_srcs
                and self.fwd_monitors == other.fwd_monitors
                and self.adj_monitors == other.adj_monitors
                and self.fom_func == other.fom_func
                and self.grad_func == other.grad_func
                and self.pos_max_freqs == other.pos_max_freqs
                and self.neg_min_freqs == other.neg_min_freqs
                and self.all_freqs == other.all_freqs
                # and self.spectral_weights == other.spectral_weights
                # and self.reduce_func == other.reduce_func
            )
        return super().__eq__(other)
    
    def __copy__(self) -> FoM:
        """Create a copy of this FoM."""
        return FoM(
            self.fom_func,
            self.grad_func,
            self.foms,
            self.weights,
            self.fwd_srcs,
            self.fwd_monitors,
            self.adj_srcs,
            self.adj_monitors,
            self.polarization,
            self.pos_max_freqs,
            self.neg_min_freqs,
            self.all_freqs,
            # self.spectral_weights,
            # self.reduce_func,
        )

    def as_dict(self) -> dict:
        """Return a dictionary representation of this FoM."""
        
        self.fom_func,
        self.grad_func,
        self.foms,
        self.weights,
        self.fwd_srcs,
        self.fwd_monitors,
        self.adj_srcs,
        self.adj_monitors,
        self.polarization,
        self.pos_max_freqs,
        self.neg_min_freqs,
        self.all_freqs,
        # self.spectral_weights,
        # self.reduce_func,
        
        data: dict[str, Any] = {}
        data['type'] = type(self).__name__
        
        if data['type'] == 'FoM':  # Generic FoM needs to copy functions
            data['fom_func'] = self.fom_func
            data['grad_func'] = self.grad_func
        data['foms'] = self.foms
        data['weights'] = self.weights
        data['fwd_srcs'] = [f['name'] for f in self.fwd_srcs]
        data['fom_monitors'] = [f['name'] for f in self.fwd_monitors]
        data['adj_srcs'] = [f['name'] for f in self.adj_srcs]
        data['grad_monitors'] = [f['name'] for f in self.adj_monitors]
        
        data['polarization'] = self.polarization
        data['pos_max_freqs'] = self.pos_max_freqs
        data['neg_min_freqs'] = self.neg_min_freqs
        data['all_freqs'] = self.all_freqs

        return data

    @staticmethod
    def from_dict(input_dict: dict) -> FoM:
        """Create a FoM from a dictionary representation."""
        data = copy(input_dict)
        fom_cls: type[FoM] = getattr(sys.modules[__name__], data.pop('type'))
        return fom_cls(**data)
    
    def reset_monitors(self):
        """Reset all of the monitors used to calculate the FoM."""
    
        if len(self.foms) > 0:
            map(FoM.reset_monitors, flatten(self.foms))
        else:
            for mon in self.fwd_monitors:
                mon.reset()
            for mon in self.adj_monitors:
                mon.reset()
    
    # Creating and linking new simulations
    
    def link_forward_sim(self, sim: Simulation):
        """Link this FoM's fwd_monitors to a provided simulation."""
        self.fwd_monitors = [sim.objects[m.name] for m in self.fwd_monitors]

    def link_adjoint_sim(self, sim: Simulation):
        """Link this FoM's adj_monitors to a provided simulation."""
        self.adj_monitors = [sim.objects[m.name] for m in self.adj_monitors]

    def create_forward_sim(
        self, base_sim: Simulation,
        link_sims:bool = True,
        ) -> list[Simulation]:
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
            if link_sims:
                for i, foms in enumerate(fwd_sim_map.values()):
                    for fom in foms:
                        fom.link_forward_sim(sims[i])
            return sims
    
    def create_adjoint_sim(
            self, base_sim: Simulation,
            link_sims:bool = True,
        ) -> list[Simulation]:
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
        if link_sims:
            for i, foms in enumerate(adj_sim_map.values()):
                for fom in foms:
                    fom.link_adjoint_sim(sims[i])
        return sims

    # Figure of Merit Function Handling
    
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
                    ({'reduce': False, **kwargs} for _ in foms),
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


    def compute_fom(self, reduce: bool = True, *args, **kwargs) -> npt.NDArray:
        """Compute the figure of merit."""
        total_fom = self.fom_func(*args, **kwargs)
        self.reset_monitors()
        # return self._subtract_neg(total_fom)
        if reduce:
            return self.reduce_func(f)
        return f
    
        # """Compute the weighted sum of the FoMs."""
        # fom_results = np.array([
        #     SuperFoM._compute_prod(
        #         FoM.compute_fom,
        #         fom_tup,
        #         *args,
        #         **kwargs,
        #     )
        #     for fom_tup in self.foms
        # ])
        # self.performance_weighting(fom_results)
        # # fom_results = np.dot(fom_results, spectral_weights).dot(performance_weights)
        # return np.einsum('i,i...->...', self.weights, fom_results)


    def compute_grad(self, apply_performance_weights=False, 
                     *args, **kwargs) -> npt.NDArray:
        """Compute the gradient of the figure of merit."""
        total_grad = self.grad_func(*args, **kwargs)
        self.reset_monitors()
        # return self._subtract_neg(total_grad)
        return np.dot(total_grad, self.spectral_weights)

        # """Compute the weighted sum of the gradients."""
        # grad_results = np.array([
        #     SuperFoM._prod_rule(
        #         fom_tup,
        #         *args,
        #         **kwargs,
        #     )
        #     for fom_tup in self.foms
        # ])
        # # grad_results = np.dot(grad_results, spectral_weights).dot(performance_weights)
        # if apply_performance_weights:
        #     assert len(self.weights)==len(self.performance_weights)
        #     return np.einsum('i,i...->...', self.weights*self.performance_weights, grad_results)
        # return np.einsum('i,i...->...', self.weights, grad_results)







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



if __name__ == '__main__':
    vipdopt.logger = setup_logger('logger', 0)
    vipdopt.lumapi = import_lumapi(
        'C:\\Program Files\\Lumerical\\v221\\api\\python\\lumapi.py'
    )
    base_sim = Simulation('test_data\\sim.json')
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