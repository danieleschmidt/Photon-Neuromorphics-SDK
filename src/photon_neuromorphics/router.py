"""
SpikeRouter: Routes optical spikes between neurons via synaptic connections.

Supports configurable topologies and manages the network simulation step.
"""

from .spike import OpticalSpike


class SpikeRouter:
    """
    Routes optical spikes through a network of neurons and synapses.

    Parameters
    ----------
    topology : str
        Routing topology identifier ('star', 'mesh', etc.)
        Currently 'star' is the primary supported topology.
    """

    def __init__(self, topology='star'):
        self.topology = topology
        self._neurons = {}          # neuron_id -> OpticalLIFNeuron
        self._connections = {}      # (src_id, dst_id) -> PhotonicSynapticWeight
        self._adjacency = {}        # src_id -> list of dst_ids
        self._output_spikes = []    # spikes at output neurons this timestep
        self._output_neurons = set()
        self._timestep = 0

    def add_neuron(self, neuron_id, neuron, is_output=False):
        """
        Register a neuron in the network.

        Parameters
        ----------
        neuron_id : str or int
            Unique identifier for this neuron
        neuron : OpticalLIFNeuron
            Neuron instance to register
        is_output : bool
            Mark this neuron as an output neuron
        """
        self._neurons[neuron_id] = neuron
        if neuron_id not in self._adjacency:
            self._adjacency[neuron_id] = []
        if is_output:
            self._output_neurons.add(neuron_id)

    def add_connection(self, src_id, dst_id, synapse):
        """
        Add a synaptic connection between two neurons.

        Parameters
        ----------
        src_id : str or int
            Source neuron identifier
        dst_id : str or int
            Destination neuron identifier
        synapse : PhotonicSynapticWeight
            Synaptic weight / MZI for this connection
        """
        self._connections[(src_id, dst_id)] = synapse
        if src_id not in self._adjacency:
            self._adjacency[src_id] = []
        if dst_id not in self._adjacency:
            self._adjacency[dst_id] = []
        self._adjacency[src_id].append(dst_id)

    def route(self, spike):
        """
        Route a spike from its source to all connected destination neurons.

        Parameters
        ----------
        spike : OpticalSpike
            The spike to route

        Returns
        -------
        list of str/int
            Destination neuron ids that received input
        """
        src_id = spike.source_id
        destinations = self._adjacency.get(src_id, [])
        reached = []

        for dst_id in destinations:
            synapse = self._connections.get((src_id, dst_id))
            if synapse is not None and dst_id in self._neurons:
                transmitted_power = synapse.apply(spike.power)
                self._neurons[dst_id].integrate(transmitted_power)
                reached.append(dst_id)

        return reached

    def step(self, inputs, dt=1.0):
        """
        Execute one simulation timestep.

        1. Apply external inputs to input neurons
        2. Check for spikes in all neurons
        3. Route spikes to connected neurons
        4. Collect output spikes

        Parameters
        ----------
        inputs : dict
            Mapping of neuron_id -> input_power for this timestep
        dt : float
            Timestep duration (ms, default 1.0)

        Returns
        -------
        list of OpticalSpike
            Spikes emitted by output neurons this timestep
        """
        self._output_spikes = []

        # Apply external inputs
        for neuron_id, power in inputs.items():
            if neuron_id in self._neurons:
                self._neurons[neuron_id].integrate(power, dt)

        # Check all neurons for spikes and route
        for neuron_id, neuron in self._neurons.items():
            if neuron.spike():
                spike = OpticalSpike(
                    wavelength_nm=1550.0,  # standard C-band wavelength
                    power_mw=1.0,
                    timestamp=self._timestep,
                    source_id=neuron_id,
                )
                # Route spike to downstream neurons
                self.route(spike)

                # Collect if output neuron
                if neuron_id in self._output_neurons:
                    self._output_spikes.append(spike)

        self._timestep += 1
        return self._output_spikes

    def get_outputs(self):
        """
        Get output spikes from the most recent timestep.

        Returns
        -------
        list of OpticalSpike
            Output spikes from the last step() call
        """
        return list(self._output_spikes)

    def reset_timestep(self):
        """Reset the internal timestep counter."""
        self._timestep = 0

    def __repr__(self):
        return (
            f"SpikeRouter(topology='{self.topology}', "
            f"neurons={len(self._neurons)}, "
            f"connections={len(self._connections)})"
        )
