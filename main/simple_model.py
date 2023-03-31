import os
from time import time as t
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.monitors import NetworkMonitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
import torch
import matplotlib.pyplot as plt
from torch import bernoulli, ones, randn
from time import time
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from torchvision import transforms
import os

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
#!echo "./data/MNIST" >> ./.gitignore ### run once only


total_timesteps = 250
dt = 1
num_epochs = 1
progress_interval = 10
update_interval = 250

intensity = 128
image_shape = (28, 28)
n_sensory_neurons = 100
n_output_neurons = n_sensory_neurons
dendrite_field_shape = (10, 10)
num_dendrite_layers = 1
input_neuron_weight = 5

seed = 8
batch_size = 1

def simple_model():
    network = Network(dt=dt, learning=True, batch_size=1, reward_fn=None)
    image_dim = image_shape[0]*image_shape[1]

    input_layer = Input(n=image_dim, traces=True)
    network.add_layer(input_layer, 'simulated input')

    sensory_layer = LIFNodes(n=n_sensory_neurons, traces=True)
    network.add_layer(sensory_layer, 'light-receptive neurons')

    output_layer = LIFNodes(n=n_output_neurons, traces=True)
    network.add_layer(output_layer, 'output neurons')

    dendrite_layers_container = []
    dendrite_dim = dendrite_field_shape[0]*dendrite_field_shape[1]

    for iii in range(num_dendrite_layers):
        dendrite_layers_container.append(Input(n=dendrite_dim, traces=True))
        network.add_layer(dendrite_layers_container[iii], f"Dendrites {iii}")
        
        input_sensory_connection = Connection(
                source=input_layer,
                target=sensory_layer,
                w = input_neuron_weight*torch.rand(input_layer.n, sensory_layer.n))
    network.add_connection(input_sensory_connection,
                           source='simulated input',
                           target='light-receptive neurons')

    sensory_output_connection = Connection(
                    source=sensory_layer,
                    target=output_layer,
                    update_rule=PostPre, nu=(1e-4, 1e-2),
                    w = 22.5*torch.diag(torch.ones(n_output_neurons))) #~10-1 connections to any given neuron
    network.add_connection(sensory_output_connection,
                           source='light-receptive neurons',
                           target='output neurons')

    dfs_dim = dendrite_field_shape[0]*dendrite_field_shape[1]
    dendrite_sensory_connections_container = []

    for iii in range(num_dendrite_layers):

        weights = torch.bernoulli(0.2*torch.ones(dendrite_layers_container[0].n, sensory_layer.n))
            #each feedback neuron connected to ~5 input neurons
        dendrite_sensory_connections_container.append(Connection(
                                source=dendrite_layers_container[iii],
                                target=sensory_layer,
                                w = weights*randn(weights.shape) ))

        network.add_connection(dendrite_sensory_connections_container[iii],
                               source=f"Dendrites {iii}",
                               target='light-receptive neurons')
    spikes = { }
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer],
                               state_vars = ["s"],
                               time = total_timesteps)
        network.add_monitor(spikes[layer], name = f"{layer} spikes")

    voltages = { }
    for layer in set(network.layers)-{"simulated input"}-set([f"Dendrites {iii}" for iii in range(num_dendrite_layers)]):
        voltages[layer] = Monitor(network.layers[layer],
                                  state_vars = ["v"],
                                  time=total_timesteps)
        network.add_monitor(voltages[layer], name = f"{layer} voltages")
    return network