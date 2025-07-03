"""Configuration manager for Dispersion Beam Splitter.
Handles functions that are unique to the DispBS configuration;
but called outside of the initial, or too complex for the, jinja initialization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

import numpy as np
import yaml
import copy
from overrides import override

from vipdopt.configuration.config import Config
from vipdopt.configuration.template import TemplateRenderer
from vipdopt.utils import ensure_path


class DispBSConfig(Config):
    """Config object specifically for use with the dispersion beam splitter optimization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._do_validation = True

    @override
    def __setitem__(self, name: str, value: Any) -> None:
        super().__setitem__(name, value)
        if self._do_validation:
            self._validate()

    @ensure_path
    @override
    def read_file(self, fname: Path, cfg_format: str = 'auto') -> None:
        super().read_file(fname, cfg_format=cfg_format)
        self._validate()

    @overload
    def update(self, __m: SupportsKeysAndGetItem, **kwargs: Any) -> None: ...

    @overload
    def update(self, __m: Iterable[tuple[Any, Any]], **kwargs) -> None: ...

    @overload
    def update(self, **kwargs: Any) -> None: ...

    def update(self, *args, **kwargs: Any) -> None:
        """Update self with values from another dictionary-like object."""
        self._do_validation = False
        if len(args) == 0:
            super().update(**kwargs)
        else:
            super().update(args[0], **kwargs)
        self._do_validation = True
        self._validate()

    def derive_params(self, renderer: TemplateRenderer|None = None):
        """Derive the parameters that depend on the config files."""
        # new_yaml = renderer.render(data=self, pi=np.pi)
        # new_params = yaml.safe_load(new_yaml)
        # self.update(new_params)
        self._derive_params()


    def _layer_gradient(self):
        voxels_per_layer = np.array([1, 2, 4, 4, 4, 4, 5, 5, 5, 6])
        assert np.sum(voxels_per_layer) == self.device_voxels_z

        if self.flip_gradient:
            voxels_per_layer = np.flip(voxels_per_layer)

        self.voxels_per_layer = np.array

    def _do_rejection(self):
        # Determine the wavelengths that will be directed to each focal area
        self.spectral_focal_plane_map = [
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
        ]

        desired_band_width_um = 0.04  # 0.08#0.12#0.18

        self.wl_per_step_um = self.lambda_values_um[1] - self.lambda_values_um[0]

        #
        # weights will be 1 at the center and drop to 0 at the edge of the band
        #

        weighting_by_band = np.zeros((self.num_bands, self.num_design_frequency_points))

        for band_idx in range(self.num_bands):
            wl_center_um = self.desired_peaks_per_band_um[band_idx]

            for wl_idx in range(self.num_design_frequency_points):
                wl_um = self.lambda_values_um[wl_idx]
                weight = -0.5 + 1.5 * np.exp(
                    -((wl_um - wl_center_um) ** 2 / (desired_band_width_um**2))
                )
                weighting_by_band[band_idx, wl_idx] = weight

        spectral_focal_plane_map_directional_weights = np.zeros((
            4,
            self.num_design_frequency_points,
        ))
        spectral_focal_plane_map_directional_weights[0, :] = weighting_by_band[0]
        spectral_focal_plane_map_directional_weights[1, :] = weighting_by_band[1]
        spectral_focal_plane_map_directional_weights[2, :] = weighting_by_band[2]
        spectral_focal_plane_map_directional_weights[3, :] = weighting_by_band[1]

        self.spectral_focal_plane_map_directional_weights = (
            spectral_focal_plane_map_directional_weights
        )

    def _wavelengths_and_angles(self, incident_angles:list, dispersion_factors:list, ):
        '''Generate set of wavelengths and angles for optimisation according to specified dispersion factor (ONE).'''

        wavelengths = self.data['lambda_values_um']
        new_angles = []

        for i_th, theta in enumerate(incident_angles):
            # calculate regular dispersive deflection angles

            # special case for off normal incidence
            if theta != 0:
                period = self.data['lambda_center_um']/np.sin(np.deg2rad(theta))
            else:
                period = np.inf

            angles = np.rad2deg(np.arcsin(wavelengths/period))

            # multiply angle difference with dispersion factor
            delta_angles = angles - theta
            spread = delta_angles * dispersion_factors[i_th]

            new_angles.append(spread + theta)

        return wavelengths, new_angles

    def _derive_params(self, *args, **kwargs):
        """Derive the parameters that depend on the config files."""
        
        
        # NOTE: Initially, the config has two FoMs that correspond to two polarizations being split.
        # NOTE: This copy-pastes each of those FoMs with different wavelengths and angles pertaining to dispersion control.
        # =================================================================================================
        wavelengths = self.data['lambda_values_um']
        foms = self.data['figures_of_merit']
        base_sim = self.data['base_simulation']

        new_foms = copy.deepcopy(foms)
        new_foms['foms'] = {}
        new_adj_srcs = {}
        new_foms['weights'] = []

        fom_counter = 0
        for fom_idx, fom_name in enumerate(foms['foms']):
            fom = foms['foms'][fom_name]
            # Identify the adjoint sources being used by the FoMs.
            fom_adj_srcs = fom['adj_srcs']
            # Get wavelengths and angles, which requires incident angles of the adjoint sources
            inc_angles = [base_sim['objects'][x]['properties']['angle theta'] for x in fom_adj_srcs]
            disp_factors = [self.data['dispersion_factors'][x] for x in fom_adj_srcs]
            _, inc_angles = self._wavelengths_and_angles(incident_angles=inc_angles,
                                                        dispersion_factors=disp_factors)

            # Duplicate adjoint source and FoM according to inc_angles
            for fom_adjsrc_idx, fom_adjsrc in enumerate(fom_adj_srcs):
                template_adjsrc = copy.deepcopy(base_sim['objects'][fom_adjsrc])
                template_fom = copy.deepcopy(fom)
                # For each angle:
                for wl_idx, inc_th in enumerate(inc_angles[fom_adjsrc_idx]):
                    
                    new_adj_src = copy.deepcopy(template_adjsrc)
                    
                    # Change name
                    new_adj_src['name'] = new_adj_src['name'] + f'_wl{wl_idx}'
                    # Adjust theta
                    new_adj_src.update({'angle theta':inc_th})
                    # Adjust x
                    new_adj_src['properties']['x'] = new_adj_src['properties']['x'] * \
                        (np.tan(np.radians(template_adjsrc['properties']['angle theta']))/np.tan(np.radians(inc_th)))
                    # Adjust wavelength
                    new_adj_src['properties']['wavelength start'] = wavelengths[wl_idx] * 1e-6
                    new_adj_src['properties']['wavelength stop'] = wavelengths[wl_idx] * 1e-6
                    new_adj_src['properties']['wavelength span'] = 0
                    new_adj_src['properties']['center wavelength'] = wavelengths[wl_idx] * 1e-6
                    
                    new_fom = copy.deepcopy(template_fom)
                    # Adjust FoM's corresponding adjoint source
                    new_fom_name = f'fom_{fom_counter}_0'          # Adjust this according to how you want the FoMs to be put together
                    new_fom['adj_srcs'] = [new_adj_src['name']]

                    new_adj_srcs.update({new_adj_src['name']: new_adj_src})
                    new_foms['foms'].update({new_fom_name: new_fom})
                    
                    new_foms['weights'].append(copy.deepcopy(foms['weights'][fom_idx]))
                    
                    fom_counter += 1

        self.data['figures_of_merit'] = new_foms
        self.data['base_simulation']['objects'].update(new_adj_srcs)
        # =================================================================================================

        # #! TODO: FIX THE BELOW
        # if self.get('border_optimization'):
        #     self.device_size_lateral_bordered_um = 2 * self.border_size_um

        #     if self.get('evaluate_bordered_extended'):
        #         self.border_size_um = self.device_size_lateral_um

        #     self.device_size_lateral_bordered_um += 2 * self.border_size_um
        #     self.device_voxels_lateral_bordered = int(
        #         np.round(
        #             self.device_size_lateral_bordered_um
        #             / self.geometry_spacing_lateral_um
        #         )
        #     )

        #     # 1 if mesh_spacing_um == 0.017
        #     self.device_voxels_simulation_mesh_lateral_bordered = (
        #         int(self.device_size_lateral_bordered_um / self.mesh_spacing_um) + 1
        #     )
        # else:
        #     self.device_voxels_simulation_mesh_lateral_bordered = (
        #         self.data['device_mesh_voxels_x']
        #     )

        # if self.get('use_airy_approximation'):
        #     self.gaussian_waist_radius_um = (
        #         self.airy_correction_factor * self.mid_lambda_um * self.f_number
        #     )
        # else:
        #     self.gaussian_waist_radius_um = self.mid_lambda_um / (
        #         np.pi * (1.0 / (2 * self.f_number))
        #     )
        # self.gaussian_waist_radius_um *= self.beam_size_multiplier

        # if self.get('sidewall_extend_pml'):
        #     self.sidewall_thickness_um = (
        #         self.fdtd_size_x_um - self.device_size_lateral_um
        #     ) / 2

        # if self.get('add_infrared'):
        #     self._add_infrared()

        # if self.get('layer_gradient'):
        #     if self.num_vertical_layers != VERTICAL_LAYERS:
        #         raise ValueError(
        #             f"Expected 'num_vertical_layers'=={VERTICAL_LAYERS},"
        #             f' got {self.num_vertical_layers}.'
        #         )
        #     self._layer_gradient()

        # if self.get('explicit_band_centering'):
        #     self._explicit_band_centering()

        # if self.get('do_rejection'):
        #     self._do_rejection()

    def _validate(self):
        """Validate the config file and compute conditional attributes."""
        if self.get('border_optimization') and self.get('use_smooth_blur'):
            msg = (
                "Combining 'border_optimization' and "
                "'use_smooth_blur' is not supported"
            )
            raise ValueError(msg)

        if self.get('border_optimization') and self.get('num_sidewalls') != 0:
            msg = (
                "Combining 'border_optimization' and "
                "'num_sidewalls' > 0 is not supported"
            )
            raise ValueError(msg)

        if self.get('add_pdaf'):
            dvl = self.get('device_voxels_lateral_um')
            if dvl is None or dvl % 2 != 0:
                raise ValueError(
                    "Expected 'device_voxels_lateral_um' to be even for"
                    ' PDAF implementation ease, got '
                    f"'{dvl}'."
                )

            if self.get('add_infrared'):
                raise ValueError("'add_pdaf and 'add_infrared' are not compatible.")

        if (
            self.get('reinterpolate_permittivity') is False
            and self.get('reinterpolate_permittivity_factor') != 1
        ):
            raise ValueError(
                "Expected 'reinterpolate_permittivity_factor' to be 1 if not"
                ' reinterpolating permittivity,'
                f" got '{self.get('reinterpolate_permittivity_factor')}'."
            )
