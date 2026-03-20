# Photon Neuromorphics SDK

A Python library for simulating **silicon-photonic spiking neural networks (SNNs)**.

Models optical neurons, MZI-based synaptic weights, spike routing, and
network-level computation — all with physically-grounded photonic primitives.

---

## Overview

Neuromorphic computing with silicon photonics combines the speed of light with
the energy efficiency of event-driven spiking computation. This library provides
building blocks for simulating such systems:

| Component | Description |
|-----------|-------------|
| `OpticalLIFNeuron` | Leaky integrate-and-fire neuron driven by optical power |
| `PhotonicSynapticWeight` | MZI-encoded weight with STDP plasticity |
| `OpticalSpike` | Optical pulse event (wavelength, power, timestamp) |
| `SpikeRouter` | Routes spikes between neurons via synaptic connections |
| `OpticalSNNDemo` | XOR demonstration network |

---

## Installation

```bash
pip install -e .
# or
pip install numpy
```

---

## Quick Start

```python
from photon_neuromorphics import (
    OpticalLIFNeuron,
    PhotonicSynapticWeight,
    OpticalSpike,
    SpikeRouter,
    OpticalSNNDemo,
)

# Create a neuron
neuron = OpticalLIFNeuron(tau_m=20.0, v_threshold=1.0, refractory_period=2)

# Integrate optical input
for t in range(30):
    neuron.integrate(input_power=0.5, dt=1.0)
    if neuron.spike():
        print(f"Spike at t={t}")

# MZI synaptic weight
syn = PhotonicSynapticWeight(phase=0.0)  # full transmission
print(f"Transmission: {syn.transmission():.3f}")  # 1.0

# Apply weight to optical power
out_power = syn.apply(input_power=2.0)  # returns 2.0

# STDP learning
syn.learn(pre_spike=True, post_spike=True, lr=0.01)

# Optical spike event
spike = OpticalSpike(
    wavelength_nm=1550.0,
    power_mw=1.0,
    timestamp=42,
    source_id='N0'
)
print(spike)

# XOR demo
demo = OpticalSNNDemo()
for inputs in [(0,0), (0,1), (1,0), (1,1)]:
    count = demo.run_xor(inputs, timesteps=50)
    print(f"XOR{inputs} → {count} spikes")
```

---

## Architecture: XOR Network

```
  I0 ──┬──(MZI w=1.0)──→ H_OR  ──(MZI w=1.0)──→ OUT
       └──(MZI w=0.75)─→ H_AND ──(MZI w=1.0)──→ INH ─(inhibit)─→ OUT
  I1 ──┬──(MZI w=1.0)──→ H_OR
       └──(MZI w=0.75)─→ H_AND
```

- **H_OR**: fires when any input is active
- **H_AND**: fires only when both inputs are active (higher threshold)
- **INH**: inhibitory interneuron that suppresses OUT when both inputs fire

---

## Physics

### Leaky Integrate-and-Fire

```
v(t+dt) = v(t) * exp(-dt/τ_m) + P_in * dt
```

Fires (emits spike) when `v >= v_threshold`, then resets to `v_reset`.

### MZI Transmission

```
T(φ) = cos²(φ/2)
```

- φ = 0 → T = 1.0 (constructive interference, full pass)
- φ = π → T = 0.0 (destructive interference, blocked)

### STDP Plasticity

```
Pre + Post spike  → φ -= lr * π  (potentiation: higher T)
Pre spike only    → φ += lr * π  (depression: lower T)
φ is clamped to [0, π]
```

---

## Running Tests

```bash
cd Photon-Neuromorphics-SDK
python -m pytest tests/ -v
```

---

## Project Structure

```
src/
  photon_neuromorphics/
    __init__.py       # public API
    neuron.py         # OpticalLIFNeuron
    synapse.py        # PhotonicSynapticWeight
    spike.py          # OpticalSpike
    router.py         # SpikeRouter
    network.py        # OpticalSNNDemo
examples/
  demo.py             # XOR demo script
tests/
  test_photon_neuromorphics.py
requirements.txt
pyproject.toml
```

---

## License

MIT
