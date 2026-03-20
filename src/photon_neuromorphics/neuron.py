"""
OpticalLIFNeuron: Leaky Integrate-and-Fire neuron with optical input.

Models a silicon-photonic spiking neuron where input is optical power
and the membrane potential evolves according to LIF dynamics.
"""

import numpy as np


class OpticalLIFNeuron:
    """
    Leaky Integrate-and-Fire neuron driven by optical input power.

    The membrane voltage evolves as:
        dv/dt = -v/tau_m + input_power

    Discretized:
        v(t+dt) = v(t) * exp(-dt/tau_m) + input_power * dt

    Parameters
    ----------
    tau_m : float
        Membrane time constant in ms (default 20.0)
    v_threshold : float
        Spike threshold voltage (default 1.0)
    v_reset : float
        Reset voltage after spike (default 0.0)
    refractory_period : int
        Number of timesteps neuron is silent after spiking (default 2)
    """

    def __init__(self, tau_m=20.0, v_threshold=1.0, v_reset=0.0, refractory_period=2):
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_period = refractory_period

        self._v = v_reset
        self._t_last_spike = -np.inf
        self._refractory_counter = 0

    def integrate(self, input_power, dt=1.0):
        """
        Update membrane potential with optical input.

        If the neuron is in refractory period, integration is blocked.
        After integration, if v >= threshold, the neuron fires automatically.

        Parameters
        ----------
        input_power : float
            Optical input power (mW or normalized units)
        dt : float
            Timestep (ms, default 1.0)
        """
        if self._refractory_counter > 0:
            self._refractory_counter -= 1
            return

        # Leaky integration: exponential decay + input
        decay = np.exp(-dt / self.tau_m)
        self._v = self._v * decay + input_power * dt

    def spike(self):
        """
        Check if neuron fires and handle spike.

        Returns
        -------
        bool
            True if membrane potential exceeds threshold.
        """
        if self._v >= self.v_threshold:
            self._t_last_spike = 0  # will be updated via external time tracking
            self._refractory_counter = self.refractory_period
            self.reset()
            return True
        return False

    def reset(self):
        """Reset membrane potential to v_reset after a spike."""
        self._v = self.v_reset

    def state(self):
        """
        Return current neuron state.

        Returns
        -------
        tuple
            (v, t_last_spike) — current membrane voltage and time of last spike
        """
        return (self._v, self._t_last_spike)

    def __repr__(self):
        v, t = self.state()
        return (
            f"OpticalLIFNeuron(v={v:.4f}, tau_m={self.tau_m}, "
            f"threshold={self.v_threshold}, refractory={self.refractory_period})"
        )
