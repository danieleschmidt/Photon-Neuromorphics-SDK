"""
Photon Neuromorphics SDK - XOR Demo

Demonstrates the OpticalSNNDemo class by running a silicon-photonic
spiking neural network on all 4 XOR input combinations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photon_neuromorphics import OpticalSNNDemo


def run_xor_demo():
    print("=" * 60)
    print("  Photon Neuromorphics SDK - Optical XOR SNN Demo")
    print("=" * 60)
    print()

    demo = OpticalSNNDemo()
    demo.build_xor_network()

    print("Network architecture:")
    print("  Input neurons:   I0, I1")
    print("  Hidden neurons:  H_OR (OR-like), H_AND (AND-like), INH (inhibitory)")
    print("  Output neuron:   OUT")
    print("  Wavelength:      1550 nm (C-band)")
    print()

    xor_cases = [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0),
    ]

    timesteps = 50
    threshold = 5

    print(f"Running {timesteps} timesteps per input pair...")
    print()
    print(f"{'Input (A,B)':<15} {'Expected':<12} {'Spike Count':<15} {'Predicted':<12} {'Correct?'}")
    print("-" * 65)

    all_correct = True
    for (a, b), expected in xor_cases:
        count = demo.run_xor((a, b), timesteps=timesteps)
        predicted = 1 if count >= threshold else 0
        correct = predicted == expected
        if not correct:
            all_correct = False
        status = "✓" if correct else "✗"
        print(f"  ({a}, {b})         {expected:<12} {count:<15} {predicted:<12} {status}")

    print()
    if all_correct:
        print("✓ All XOR cases correctly classified!")
    else:
        print("⚠ Some XOR cases misclassified (check network tuning)")

    print()
    print("Spike trains for (0,1) input:")
    demo2 = OpticalSNNDemo()
    demo2.build_xor_network()
    spike_train = []
    for t in range(20):
        spikes = demo2.router.step({'I1': 0.6}, dt=1.0)
        spike_train.append(len(spikes))
    print(" ".join(str(s) for s in spike_train))
    print(f"Total output spikes (0,1) first 20 steps: {sum(spike_train)}")

    print()
    print("Done.")


if __name__ == '__main__':
    run_xor_demo()
