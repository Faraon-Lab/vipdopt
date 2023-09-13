"""Configuration manager for SonyBayerFilter."""

from overrides import override

from vipdopt.configuration.config import Config


class SonyBayerConfig(Config):
    """Config object specifically for use with the Sony bayer filter optimization."""

    # List of all the cascading parameters, their dependencies, and a function to
    # compute it.
    # TODO: Is this going to have to be hard-coded?
    _cascading_parameters = (
        # e.g. ('sum_of_a_and_b', ('a', 'b'), lambda a, b: a + b),
        (
            'total_iterations',
            ('num_epochs', 'num_iterations_per_epoch'),
            lambda a, b: a * b
        ),
    )

    def __init__(self):
        """Initialize SonyBayerConfig object."""
        super().__init__()

    def _derive_cascading_params(self):
        """Compute all the cascading parameters."""
        for (name, deps, f) in self._cascading_parameters:
            # Map all dependency attributes to their values
            deps_vals = tuple(self.__getattr__(x) for x in deps)

            # Call the provided function to compute `name` with the dependencies
            self.__setattr__(name, f(*deps_vals))

    @override
    def read_file(self, filename: str, cfg_format: str = 'yaml'):
        super().read_file(filename, cfg_format)
        self._derive_cascading_params()
