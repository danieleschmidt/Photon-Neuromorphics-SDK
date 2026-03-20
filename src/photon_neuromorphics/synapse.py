"""
PhotonicSynapticWeight: MZI-based synaptic weight using phase encoding.

A Mach-Zehnder Interferometer (MZI) modulates optical transmission based on
the phase difference between its two arms. The transmission is:
    T = cos²(phase / 2)

STDP-like plasticity updates the phase based on pre/post-synaptic spike timing.
"""

import numpy as np


class PhotonicSynapticWeight:
    """
    Synaptic weight encoded as MZI phase.

    The optical transmission through an MZI is:
        T(phase) = cos²(phase / 2)

    Phase 0       → T = 1.0  (full transmission)
    Phase π       → T = 0.0  (zero transmission)
    Phase π/2     → T = 0.5  (half transmission)

    Parameters
    ----------
    phase : float
        Initial MZI phase in radians (default 0.0)
    """

    def __init__(self, phase=0.0):
        self._phase = phase

    def transmission(self):
        """
        Return optical transmission for current phase.

        Returns
        -------
        float
            Transmission in [0, 1] = cos²(phase/2)
        """
        return np.cos(self._phase / 2) ** 2

    def set_phase(self, phase):
        """
        Update MZI phase.

        Parameters
        ----------
        phase : float
            New phase in radians
        """
        self._phase = phase

    def apply(self, input_power):
        """
        Apply synaptic weight to optical input.

        Parameters
        ----------
        input_power : float
            Input optical power

        Returns
        -------
        float
            Transmitted optical power = input_power * T(phase)
        """
        return input_power * self.transmission()

    def learn(self, pre_spike, post_spike, lr=0.01):
        """
        STDP-like Hebbian plasticity rule.

        If pre and post fire together → potentiation (reduce phase → increase T)
        If pre fires but post doesn't → depression (increase phase → reduce T)
        If neither fires → no change

        Parameters
        ----------
        pre_spike : bool
            Whether the pre-synaptic neuron spiked
        post_spike : bool
            Whether the post-synaptic neuron spiked
        lr : float
            Learning rate (default 0.01)
        """
        if pre_spike and post_spike:
            # Potentiation: phase decreases → higher transmission
            self._phase -= lr * np.pi
        elif pre_spike and not post_spike:
            # Depression: phase increases → lower transmission
            self._phase += lr * np.pi

        # Clamp phase to [0, pi] for valid MZI operation
        self._phase = np.clip(self._phase, 0.0, np.pi)

    @property
    def phase(self):
        """Current MZI phase in radians."""
        return self._phase

    def __repr__(self):
        return (
            f"PhotonicSynapticWeight(phase={self._phase:.4f}, "
            f"transmission={self.transmission():.4f})"
        )
