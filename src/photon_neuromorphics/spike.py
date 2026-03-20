"""
OpticalSpike: Represents a single optical spike event.

An optical spike is a pulse of light with defined wavelength, power,
timing, and source neuron identifier.
"""


class OpticalSpike:
    """
    An optical spike pulse event.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength of the optical carrier (nm)
    power_mw : float
        Peak optical power of the spike (mW)
    timestamp : float
        Time of the spike event (ms or timestep index)
    source_id : str or int
        Identifier of the source neuron
    """

    def __init__(self, wavelength_nm, power_mw, timestamp, source_id):
        self._wavelength_nm = wavelength_nm
        self._power_mw = power_mw
        self._timestamp = timestamp
        self._source_id = source_id

    @property
    def wavelength(self):
        """Wavelength of the optical carrier in nm."""
        return self._wavelength_nm

    @property
    def power(self):
        """Peak optical power in mW."""
        return self._power_mw

    @property
    def timestamp(self):
        """Time of the spike event."""
        return self._timestamp

    @property
    def source_id(self):
        """Source neuron identifier."""
        return self._source_id

    def __repr__(self):
        return (
            f"OpticalSpike(source={self._source_id}, "
            f"t={self._timestamp}, "
            f"λ={self._wavelength_nm}nm, "
            f"P={self._power_mw}mW)"
        )

    def __eq__(self, other):
        if not isinstance(other, OpticalSpike):
            return False
        return (
            self._wavelength_nm == other._wavelength_nm
            and self._power_mw == other._power_mw
            and self._timestamp == other._timestamp
            and self._source_id == other._source_id
        )
