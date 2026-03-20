"""
photon_neuromorphics - Silicon-Photonic Spiking Neural Network Library

A Python library for simulating silicon-photonic spiking neural networks (SNNs)
using optical integrate-and-fire neurons, MZI-based synaptic weights, and
spike routing.

Classes
-------
OpticalLIFNeuron
    Leaky integrate-and-fire neuron with optical input
PhotonicSynapticWeight
    MZI-encoded synaptic weight with STDP plasticity
OpticalSpike
    Optical spike pulse event
SpikeRouter
    Network router for spike-based optical computation
OpticalSNNDemo
    Demonstration XOR spiking neural network
"""

from .neuron import OpticalLIFNeuron
from .synapse import PhotonicSynapticWeight
from .spike import OpticalSpike
from .router import SpikeRouter
from .network import OpticalSNNDemo

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "OpticalLIFNeuron",
    "PhotonicSynapticWeight",
    "OpticalSpike",
    "SpikeRouter",
    "OpticalSNNDemo",
]
