"""Module for representing sources in Lumerical simulations."""

from vipdopt.simulation.simobject import LumericalSimObject, LumericalSimObjectType


# TODO: Create an __eq__ function and also rework what a source stores. Basically
# just needs to be the name of the source in the base_sim, or some unique identifier.
# Could also just use the source names instead I suppose...
class Source(LumericalSimObject):
    """Representation of a Lumerical source object."""

    def __init__(self, name: str, obj_type: LumericalSimObjectType) -> None:
        super().__init__(name, obj_type)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Source):
            return self.name == __value.name and type(self) == type(__value)
        return super().__eq__(__value)


class DipoleSource(Source):
    def __init__(self, name: str):
        super().__init__(name, LumericalSimObjectType.DIPOLE)


class TFSFSource(Source):
    def __init__(self, name: str):
        super().__init__(name, LumericalSimObjectType.TFSF)


class GaussianSource(Source):
    def __init__(self, name: str):
        super().__init__(name, LumericalSimObjectType.GAUSSIAN)


class ForwardSource(Source):
    """Source object used in forward simulations."""

    def __init__(self, src_object, polarization=None, monitor_object=None, sim_id=None):
        """Initialize a ForwardSource object."""
        super().__init__(src_object, monitor_object=monitor_object, sim_id=sim_id)

        self.Ex_fwd = None
        self.Ey_fwd = None
        self.Ez_fwd = None
        self.Hx_fwd = None
        self.Hy_fwd = None
        self.Hz_fwd = None

        self.polarization = polarization

        # An attribute to store a temporary file name to be used in parallel processing.
        self.temp_file_name = None


class AdjointSource(Source):
    """Source object used in simulations for the adjoint method."""

    def __init__(self, src_object, polarization=None, monitor_object=None, sim_id=None):
        """Initialize a AdjointSource object."""
        super().__init__(src_object, monitor_object=monitor_object, sim_id=sim_id)

        self.Ex_adj = None
        self.Ey_adj = None
        self.Ez_adj = None
        self.Hx_adj = None
        self.Hy_adj = None
        self.Hz_adj = None

        self.polarization = polarization

        # An attribute to store a temporary file name to be used in parallel processing.
        self.temp_file_name = None
