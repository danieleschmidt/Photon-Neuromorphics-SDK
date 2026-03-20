"""
OpticalSNNDemo: Demonstration optical spiking neural network.

Implements a 2-input XOR function using a 3-layer photonic SNN:
  - Input layer: 2 input neurons (A, B)
  - Hidden layer: 2 neurons (H1=A XOR B via NAND-like, H2=AND detect)
  - Output layer: 1 output neuron

XOR truth table:
  A=0, B=0 → 0
  A=0, B=1 → 1
  A=1, B=0 → 1
  A=1, B=1 → 0
"""

import numpy as np

from .neuron import OpticalLIFNeuron
from .synapse import PhotonicSynapticWeight
from .router import SpikeRouter


class OpticalSNNDemo:
    """
    Optical SNN demonstration network implementing XOR logic.

    Architecture:
        Input neurons:  I0, I1
        Hidden neurons: H0 (OR-like), H1 (AND-like)
        Output neuron:  OUT

    XOR = OR(I0, I1) AND NOT(AND(I0, I1))

    The network uses tuned MZI weights and threshold settings to
    approximate XOR behavior over multiple timesteps.
    """

    def __init__(self):
        self.router = None

    def build_xor_network(self):
        """
        Build a 2-input XOR optical SNN.

        Creates neurons and synaptic connections tuned to perform
        XOR logic via spike rate coding.

        Returns
        -------
        SpikeRouter
            Configured network router
        """
        self.router = SpikeRouter(topology='star')

        # --- Neurons ---
        # Input neurons (low threshold so they fire easily with input)
        i0 = OpticalLIFNeuron(tau_m=5.0, v_threshold=0.5, v_reset=0.0, refractory_period=1)
        i1 = OpticalLIFNeuron(tau_m=5.0, v_threshold=0.5, v_reset=0.0, refractory_period=1)

        # Hidden: OR-like neuron — fires if any input arrives
        h_or = OpticalLIFNeuron(tau_m=10.0, v_threshold=0.4, v_reset=0.0, refractory_period=1)

        # Hidden: AND-like neuron — fires only if both inputs arrive (higher threshold)
        h_and = OpticalLIFNeuron(tau_m=10.0, v_threshold=0.9, v_reset=0.0, refractory_period=1)

        # Output: fires if OR fires but AND doesn't suppress it
        out = OpticalLIFNeuron(tau_m=10.0, v_threshold=0.4, v_reset=0.0, refractory_period=1)

        self.router.add_neuron('I0', i0)
        self.router.add_neuron('I1', i1)
        self.router.add_neuron('H_OR', h_or)
        self.router.add_neuron('H_AND', h_and)
        self.router.add_neuron('OUT', out, is_output=True)

        # --- Synapses: Input → OR (excitatory, high transmission) ---
        # phase=0 → T=1.0 (full transmission)
        self.router.add_connection('I0', 'H_OR', PhotonicSynapticWeight(phase=0.0))
        self.router.add_connection('I1', 'H_OR', PhotonicSynapticWeight(phase=0.0))

        # --- Synapses: Input → AND (excitatory, moderate) ---
        # phase=pi/3 → T=cos²(pi/6) ≈ 0.75
        self.router.add_connection('I0', 'H_AND', PhotonicSynapticWeight(phase=np.pi / 3))
        self.router.add_connection('I1', 'H_AND', PhotonicSynapticWeight(phase=np.pi / 3))

        # --- Synapses: OR → OUT (excitatory) ---
        self.router.add_connection('H_OR', 'OUT', PhotonicSynapticWeight(phase=0.0))

        # --- Synapses: AND → OUT (inhibitory via high phase = low T but conceptual suppression)
        # We implement inhibition by giving AND→OUT a subthreshold contribution
        # that partially cancels OR→OUT when AND fires. We use phase=2pi/3 → T≈0.25
        # and feed the AND output back as negative by making the output threshold higher
        # when AND fires. Here we simulate by having AND drive a "veto" on OUT:
        # Instead, we'll wire AND→OUT with phase=pi (T=0) as inhibitory placeholder,
        # and instead handle XOR by adjusting OR neuron to not fire when AND fires.
        # Practical approach: route AND→OUT with negative effective weight modeled as
        # increasing OUT's charge drain. We accomplish this by having AND→OUT use
        # phase=0 but drive a separate inhibitory interneuron that resets OUT.

        # Simpler: wire AND directly to OUT with inhibitory effect
        # We'll handle by making AND output reset OUT (via high competing input)
        # For a clean demo we'll add an inhibitory interneuron
        inh = OpticalLIFNeuron(tau_m=5.0, v_threshold=0.3, v_reset=0.0, refractory_period=1)
        self.router.add_neuron('INH', inh)

        # AND → INH (excitatory, fires inhibitory neuron)
        self.router.add_connection('H_AND', 'INH', PhotonicSynapticWeight(phase=0.0))

        # INH → OUT: we model inhibition as negative via the OUT neuron absorbing
        # a large "reset" input. Since LIF doesn't natively support negative inputs
        # in this simple model, we'll just not wire INH→OUT but instead tune thresholds
        # so that when both I0 and I1 are active, the AND neuron fires and absorbs
        # the OR output before it reaches threshold at OUT.

        # Final clean wiring:
        # When only one input active: OR fires, AND does not → OUT fires (XOR=1)
        # When both inputs active: OR fires AND AND fires → both drive OUT but
        # the key is the inhibitory path must suppress OUT.
        # We wire INH→OUT with large phase (near pi) effectively cutting transmission.
        # Instead let's use a proper approach: INH drives OUT with NEGATIVE effective weight
        # by wiring INH→OUT with a synapse where the signal goes to a "drain" representation.
        # Simplest model: INH fires → resets OUT membrane via a large competing current
        # implemented as calling OUT.reset() directly from the router. We'll subclass/monkey
        # patch... actually, let's just wire INH to OUT but have OUT threshold tuned such
        # that OR alone barely pushes it over, and AND+OR causes INH to fire which
        # pushes OUT back under. We accomplish this by making INH→OUT a "negative" add
        # by storing the INH→OUT synapse with special handling.

        # Clean implementation: use the router's step to check INH spike and reset OUT
        # Store reference for use in run_xor
        self._inh_id = 'INH'
        self._out_id = 'OUT'

        return self.router

    def run_xor(self, inputs, timesteps=50):
        """
        Simulate XOR for given binary inputs over multiple timesteps.

        Parameters
        ----------
        inputs : tuple of (int, int)
            Binary input values (A, B), each 0 or 1
        timesteps : int
            Number of simulation timesteps (default 50)

        Returns
        -------
        int
            Output spike count (higher = logic 1, lower = logic 0)
        """
        if self.router is None:
            self.build_xor_network()

        # Rebuild network fresh for each run
        self.build_xor_network()

        a, b = inputs
        input_power = 0.6  # optical power per active input

        output_count = 0
        inh_spikes_this_run = []

        for t in range(timesteps):
            step_inputs = {}
            if a:
                step_inputs['I0'] = input_power
            if b:
                step_inputs['I1'] = input_power

            # Check INH neuron state before step
            inh_neuron = self.router._neurons.get('INH')
            out_neuron = self.router._neurons.get('OUT')

            output_spikes = self.router.step(step_inputs)

            # Inhibitory suppression: if INH fired, reset OUT
            if inh_neuron is not None and out_neuron is not None:
                # Check if INH fired this step (it was routed by step())
                # We detect INH firing by monitoring the router's timestep
                # Instead, check: after step, if AND fired → suppress OUT
                # We'll check via AND neuron's refractory state
                h_and = self.router._neurons.get('H_AND')
                if h_and is not None and h_and._refractory_counter > 0:
                    # AND fired this step → suppress output
                    out_neuron.reset()
                    # Remove any output spikes this timestep
                    output_spikes = [s for s in output_spikes
                                     if s.source_id != self._out_id]

            output_count += len(output_spikes)

        return output_count

    def predict_xor(self, inputs, timesteps=50, threshold=5):
        """
        Predict binary XOR output based on spike count.

        Parameters
        ----------
        inputs : tuple of (int, int)
        timesteps : int
        threshold : int
            Spike count threshold to classify as logic 1

        Returns
        -------
        int
            0 or 1
        """
        count = self.run_xor(inputs, timesteps)
        return 1 if count >= threshold else 0
