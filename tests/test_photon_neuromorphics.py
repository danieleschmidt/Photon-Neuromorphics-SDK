"""
Tests for the photon_neuromorphics library.

Tests cover:
- OpticalLIFNeuron: init, integrate, spike, reset, refractory
- PhotonicSynapticWeight: transmission, apply, learn
- OpticalSpike: creation, properties
- SpikeRouter: add, route, step
- OpticalSNNDemo: XOR network output
"""

import sys
import os
import math

import numpy as np
import pytest

# Make sure the src/ package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photon_neuromorphics import (
    OpticalLIFNeuron,
    PhotonicSynapticWeight,
    OpticalSpike,
    SpikeRouter,
    OpticalSNNDemo,
)


# ─── OpticalLIFNeuron tests ───────────────────────────────────────────────────

def test_lif_neuron_init():
    """Neuron initializes with correct default parameters."""
    neuron = OpticalLIFNeuron()
    assert neuron.tau_m == 20.0
    assert neuron.v_threshold == 1.0
    assert neuron.v_reset == 0.0
    assert neuron.refractory_period == 2
    v, t_last = neuron.state()
    assert v == 0.0


def test_lif_neuron_init_custom():
    """Neuron accepts custom parameters."""
    neuron = OpticalLIFNeuron(tau_m=10.0, v_threshold=0.5, v_reset=0.1, refractory_period=3)
    assert neuron.tau_m == 10.0
    assert neuron.v_threshold == 0.5
    assert neuron.v_reset == 0.1
    assert neuron.refractory_period == 3


def test_lif_neuron_integrate():
    """Integrate increases membrane potential with positive input."""
    neuron = OpticalLIFNeuron(tau_m=100.0, v_threshold=10.0)
    v0, _ = neuron.state()
    neuron.integrate(input_power=1.0, dt=1.0)
    v1, _ = neuron.state()
    assert v1 > v0, "Voltage should increase with positive input"


def test_lif_neuron_integrate_decay():
    """Without input, voltage decays toward zero."""
    neuron = OpticalLIFNeuron(tau_m=5.0, v_threshold=10.0)
    # Manually set voltage
    neuron._v = 1.0
    neuron.integrate(input_power=0.0, dt=1.0)
    v, _ = neuron.state()
    assert v < 1.0, "Voltage should decay without input"
    assert v > 0.0, "Voltage should not go below zero with exponential decay"


def test_lif_neuron_spike():
    """Neuron spikes when membrane potential exceeds threshold."""
    neuron = OpticalLIFNeuron(tau_m=100.0, v_threshold=0.5, v_reset=0.0)
    neuron._v = 0.6  # above threshold
    fired = neuron.spike()
    assert fired is True


def test_lif_neuron_no_spike_below_threshold():
    """Neuron does not spike below threshold."""
    neuron = OpticalLIFNeuron(tau_m=100.0, v_threshold=1.0)
    neuron._v = 0.5  # below threshold
    fired = neuron.spike()
    assert fired is False


def test_lif_neuron_reset():
    """Reset sets membrane potential to v_reset."""
    neuron = OpticalLIFNeuron(v_reset=0.1)
    neuron._v = 2.0
    neuron.reset()
    v, _ = neuron.state()
    assert v == 0.1


def test_lif_refractory_period():
    """Neuron is silent during refractory period after spiking."""
    neuron = OpticalLIFNeuron(
        tau_m=100.0, v_threshold=0.5, v_reset=0.0, refractory_period=3
    )
    neuron._v = 0.6
    fired = neuron.spike()
    assert fired is True

    # During refractory period, integrate should not change voltage
    for _ in range(3):
        neuron.integrate(input_power=5.0, dt=1.0)
        v, _ = neuron.state()
        assert v == 0.0, "Voltage should stay at reset during refractory period"

    # After refractory, integrating should increase voltage
    neuron.integrate(input_power=5.0, dt=1.0)
    v, _ = neuron.state()
    assert v > 0.0, "Voltage should increase after refractory period ends"


# ─── PhotonicSynapticWeight tests ─────────────────────────────────────────────

def test_synaptic_weight_transmission():
    """MZI transmission is cos²(phase/2)."""
    # Phase 0 → full transmission
    syn = PhotonicSynapticWeight(phase=0.0)
    assert abs(syn.transmission() - 1.0) < 1e-9

    # Phase π → zero transmission
    syn_pi = PhotonicSynapticWeight(phase=math.pi)
    assert abs(syn_pi.transmission() - 0.0) < 1e-9

    # Phase π/2 → 0.5
    syn_half = PhotonicSynapticWeight(phase=math.pi / 2)
    assert abs(syn_half.transmission() - 0.5) < 1e-9


def test_synaptic_weight_set_phase():
    """set_phase updates the MZI phase."""
    syn = PhotonicSynapticWeight(phase=0.0)
    syn.set_phase(math.pi)
    assert abs(syn.phase - math.pi) < 1e-9
    assert abs(syn.transmission() - 0.0) < 1e-9


def test_synaptic_weight_apply():
    """apply returns input_power * transmission."""
    syn = PhotonicSynapticWeight(phase=0.0)  # T=1.0
    assert abs(syn.apply(2.0) - 2.0) < 1e-9

    syn2 = PhotonicSynapticWeight(phase=math.pi / 2)  # T=0.5
    assert abs(syn2.apply(2.0) - 1.0) < 1e-9


def test_synaptic_weight_learn():
    """STDP learning updates phase correctly."""
    syn = PhotonicSynapticWeight(phase=math.pi / 2)  # T=0.5

    initial_phase = syn.phase

    # Potentiation: pre and post both spike → phase decreases
    syn.learn(pre_spike=True, post_spike=True, lr=0.1)
    assert syn.phase < initial_phase, "Potentiation should decrease phase"

    # Reset
    syn.set_phase(math.pi / 2)

    # Depression: only pre spikes → phase increases
    syn.learn(pre_spike=True, post_spike=False, lr=0.1)
    assert syn.phase > initial_phase, "Depression should increase phase"

    # No change: neither spikes
    syn.set_phase(math.pi / 2)
    before = syn.phase
    syn.learn(pre_spike=False, post_spike=False, lr=0.1)
    assert syn.phase == before, "No learning when neither neuron fires"


def test_synaptic_weight_phase_clamped():
    """Phase is clamped to [0, pi] after learning."""
    syn = PhotonicSynapticWeight(phase=0.01)
    # Potentiation repeatedly → would go negative
    for _ in range(100):
        syn.learn(pre_spike=True, post_spike=True, lr=0.1)
    assert syn.phase >= 0.0

    syn2 = PhotonicSynapticWeight(phase=math.pi - 0.01)
    for _ in range(100):
        syn2.learn(pre_spike=True, post_spike=False, lr=0.1)
    assert syn2.phase <= math.pi


# ─── OpticalSpike tests ───────────────────────────────────────────────────────

def test_optical_spike_creation():
    """OpticalSpike stores all attributes correctly."""
    spike = OpticalSpike(
        wavelength_nm=1550.0,
        power_mw=1.0,
        timestamp=42,
        source_id='N0'
    )
    assert spike.wavelength == 1550.0
    assert spike.power == 1.0
    assert spike.timestamp == 42
    assert spike.source_id == 'N0'


def test_optical_spike_repr():
    """OpticalSpike has a meaningful string representation."""
    spike = OpticalSpike(1550.0, 1.0, 0, 'N0')
    r = repr(spike)
    assert 'N0' in r
    assert '1550' in r


# ─── SpikeRouter tests ────────────────────────────────────────────────────────

def test_spike_router_add():
    """SpikeRouter registers neurons and connections."""
    router = SpikeRouter()
    n0 = OpticalLIFNeuron()
    n1 = OpticalLIFNeuron()
    syn = PhotonicSynapticWeight()

    router.add_neuron('N0', n0)
    router.add_neuron('N1', n1)
    router.add_connection('N0', 'N1', syn)

    assert 'N0' in router._neurons
    assert 'N1' in router._neurons
    assert ('N0', 'N1') in router._connections


def test_spike_router_route():
    """route() delivers spike power to connected neurons."""
    router = SpikeRouter()
    n0 = OpticalLIFNeuron(tau_m=100.0, v_threshold=10.0)
    n1 = OpticalLIFNeuron(tau_m=100.0, v_threshold=10.0)
    syn = PhotonicSynapticWeight(phase=0.0)  # T=1.0

    router.add_neuron('N0', n0)
    router.add_neuron('N1', n1)
    router.add_connection('N0', 'N1', syn)

    spike = OpticalSpike(1550.0, 2.0, 0, 'N0')
    v_before, _ = n1.state()
    router.route(spike)
    v_after, _ = n1.state()

    assert v_after > v_before, "Routed spike should increase downstream voltage"


def test_spike_router_step():
    """step() drives inputs, fires neurons, and returns output spikes."""
    router = SpikeRouter()

    # Low threshold so one step fires it
    n0 = OpticalLIFNeuron(tau_m=100.0, v_threshold=0.1, v_reset=0.0, refractory_period=1)
    router.add_neuron('N0', n0, is_output=True)

    # Drive with enough power to cross threshold in one step
    spikes = router.step({'N0': 5.0}, dt=1.0)
    assert len(spikes) >= 1, "Output neuron should fire with sufficient input"
    assert spikes[0].source_id == 'N0'


def test_spike_router_step_no_fire():
    """step() with sub-threshold input produces no output spikes."""
    router = SpikeRouter()
    n0 = OpticalLIFNeuron(tau_m=100.0, v_threshold=100.0)
    router.add_neuron('N0', n0, is_output=True)

    spikes = router.step({'N0': 0.001}, dt=1.0)
    assert len(spikes) == 0


# ─── OpticalSNNDemo / XOR tests ───────────────────────────────────────────────

def test_xor_network_output():
    """
    XOR network produces correct spike-rate logic:
    - (0,0) → low spike count (logic 0)
    - (0,1) → high spike count (logic 1)
    - (1,0) → high spike count (logic 1)
    - (1,1) → low spike count (logic 0)
    """
    demo = OpticalSNNDemo()

    count_00 = demo.run_xor((0, 0), timesteps=50)
    count_01 = demo.run_xor((0, 1), timesteps=50)
    count_10 = demo.run_xor((1, 0), timesteps=50)
    count_11 = demo.run_xor((1, 1), timesteps=50)

    # (0,1) and (1,0) should produce more spikes than (0,0) and (1,1)
    assert count_01 > count_00, f"(0,1) should fire more than (0,0): {count_01} vs {count_00}"
    assert count_10 > count_00, f"(1,0) should fire more than (0,0): {count_10} vs {count_00}"
    assert count_01 > count_11, f"(0,1) should fire more than (1,1): {count_01} vs {count_11}"
    assert count_10 > count_11, f"(1,0) should fire more than (1,1): {count_10} vs {count_11}"
