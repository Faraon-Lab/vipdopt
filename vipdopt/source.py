"""Module for representing sources in Lumerical simulations."""
import pickle


class Source:
    """Representation of a Lumerical source object.

    Attributes:
        src_dict (dict): The source data, from the corresponding simulation object or
            dictionary containing parameters and values.
        monitor_dict (dict): The monitor at which the source is located. May be a point
            or a surface. Same type as src_dict.
        simulation (): The ID of the simulation that this source is from.
    """

    def __init__(self, src_object, monitor_object=None, sim_id=None):
        """Initialize a Source object."""
        self.src_dict = src_object
        self.monitor_dict = monitor_object
        self.simulation = sim_id

    def save_to_pickle(self, fname):
        """Pickles this Source to a file."""
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

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
