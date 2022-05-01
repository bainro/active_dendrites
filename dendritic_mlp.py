import abc
from collections.abc import Iterable
from collections import namedtuple
from typing import Optional
import numpy as np
import torch
from torch import nn
from k_winners import KWinners
from sparse_weights import SparseWeights, rezero_weights
from dendrite_segments import DendriteSegments

dendrite_output = namedtuple("dendrite_output", ["values", "indices"])
dendrite_output.__doc__ = """
A named tuple for outputs modified by `apply_dendrites`_.
:attr values: output tensor after being modulated by dendrite activations
:attr indices: the indices of the winning segments used to modulate the output tensor
.. _apply_dendrites: nupic.research.frameworks.dendrites.functional.apply_dendrites
"""


@torch.jit.script
def dendritic_bias_1d(y, dendrite_activations):
    """
    Returns the sum of the feedforward output and the max of the dendrite
    activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments respectively.
    """
    # Take max along each segment.
    winning_activations, indices = dendrite_activations.max(dim=2)
    return dendrite_output(y + winning_activations, indices)

@torch.jit.script
def gather_activations(dendrite_activations, indices):
    """
    Gathers dendritic activations from the given indices.
    :param indices: tensor of indices of winning segments;
                    shape of batch_size x num_units
    :param indices: tensor of dendritic activations;
                    shape of batch_size x num_units x num_segments
    """
    unsqueezed = indices.unsqueeze(dim=2)
    dendrite_activations = torch.gather(dendrite_activations, dim=2, index=unsqueezed)
    dendrite_activations = dendrite_activations.squeeze(dim=2)
    return dendrite_activations

@torch.jit.script
def dendritic_absolute_max_gate_1d(y, dendrite_activations):
    """
    Returns the product of the feedforward output and the sigmoid of the
    absolute max of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments, respectively.
    """
    indices = dendrite_activations.abs().max(dim=2).indices
    # Select winner by max activations, or use given indices as winners.
    if indices is None:
        winning_activations, indices = dendrite_activations.max(dim=2)
    else:
        winning_activations = gather_activations(dendrite_activations, indices)

    # Multiple by the sigmoid of the max along each segment.
    return dendrite_output(y * torch.sigmoid(winning_activations), indices)

@torch.jit.script
def dendritic_absolute_max_gate_2d(y, dendrite_activations):
    """
    Returns the output of the absolute max gating convolutional dendritic layer by
    multiplying all values in each output channel by the selected dendrite
    activations. Dendrite activations are selected based on the absolute maximum
    activations (keeping the sign) across all segments for each channel. Each
    channel has its own set of dendritic weights, and the selected activation is
    based on the the absolute max value.
    :param y: output of the convolution operation (a torch tensor with shape
              (b, c, h, w) where the axes represent the batch, channel, height, and
              width dimensions respectively)
    :param dendrite_activations: the dendrite activation values (a torch tensor
                                 with shape (b, c, d) where the axes represent the
                                 batch size, number of channels, and number of segments
                                 respectively)
    """
    indices = dendrite_activations.abs().max(dim=2).indices
    
    if indices is None:
        winning_activations, indices = dendrite_activations.max(dim=2)
    else:
        winning_activations = gather_activations(dendrite_activations, indices)

    # The following operation uses `torch.einsum` to multiply each channel by a
    # single scalar value
    #    * b => the batch dimension
    #    * i => the channel dimension
    #    * jk => the width and height dimensions

    sigmoid_activations = torch.sigmoid(winning_activations)
    y_gated = torch.einsum("bijk,bi->bijk", y, sigmoid_activations)
    return dendrite_output(y_gated, indices)

class ApplyDendritesBase(torch.nn.Module):
    """
    Base class for identifying an apply-dendrites module via `isinstance`.
    """
    pass

class DendriticBias1d(ApplyDendritesBase):
    def forward(self, y, dendrite_activations):
        return dendritic_bias_1d(y, dendrite_activations)

class DendriticGate1d(ApplyDendritesBase):
    def forward(self, y, dendrite_activations):
        return dendritic_gate_1d(y, dendrite_activations)

class DendriticAbsoluteMaxGate1d(ApplyDendritesBase):
    def forward(self, y, dendrite_activations):
        return dendritic_absolute_max_gate_1d(y, dendrite_activations)

class DendriticGate2d(ApplyDendritesBase):
    def forward(self, y, dendrite_activations):
        return dendritic_gate_2d(y, dendrite_activations)

class DendriticAbsoluteMaxGate2d(ApplyDendritesBase):
    def forward(self, y, dendrite_activations):
        return dendritic_absolute_max_gate_2d(y, dendrite_activations)

class DendriticLayerBase(SparseWeights, metaclass=abc.ABCMeta):
    """
    Base class for all Dendritic Layer modules.
    This combines a DendriteSegments module with a SparseLinear module.
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        TODO: specify the type - what is module_sparsity type?
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.dim_context = dim_context
        self.segments = None
        super().__init__(
            module,
            sparsity=module_sparsity,
            allow_extremes=True
        )

        self.segments = DendriteSegments(
            num_units=module.weight.shape[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)

    @property
    def segment_weights(self):
        return self.segments.weights

class AbsoluteMaxGatingDendriticLayer(DendriticLayerBase):
    """
    This layer is similar to `GatingDendriticLayer`, but selects dendrite activations
    based on absolute max activation values instead of just max activation values. For
    example, if choosing between activations -7.4, and 6.5 for a particular unit, -7.4
    will be chosen, and its sign will be kept.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_absolute_max_gate = DendriticAbsoluteMaxGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_absolute_max_gate(y, dendrite_activations).values
    
class AbsoluteMaxGatingDendriticLayer2d(DendriticLayer2dBase):
    """Conv version of `AbsoluteMaxGatingDendriticLayer`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_absolute_max_gate = DendriticAbsoluteMaxGate2d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_absolute_max_gate(y, dendrite_activations).values

class DendriticMLP(nn.Module):
    """
    A simple but restricted MLP with two hidden layers of the same size. Each hidden
    layer contains units with dendrites. Dendrite segments receive context directly as
    input.  The class is used to experiment with different dendritic weight
    initializations and learning parameters
    :param input_size: size of the input to the network
    :param output_size: the number of units in the output layer. Must be either an
                        integer if there is a single output head, or an iterable
                        of integers if there are multiple output heads.
    :param hidden_sizes: the number of units in each hidden layer
    :param num_segments: the number of dendritic segments that each hidden unit has
    :param dim_context: the size of the context input to the network
    :param kw: whether to apply k-Winners to the outputs of each hidden layer
    :param kw_percent_on: percent of hidden units activated by K-winners. If 0, use ReLU
    :param context_percent_on: percent of non-zero units in the context input.
    :param dendrite_weight_sparsity: the sparsity level of dendritic weights.
    :param weight_sparsity: the sparsity level of feed-forward weights.
    :param weight_init: the initialization applied to feed-forward weights; must be
                        either "kaiming" (for Kaiming Uniform) of "modified" (for
                        sparse Kaiming Uniform)
    :param dendrite_init: the initialization applied to dendritic weights; similar to
                          `weight_init`
    :param freeze_dendrites: whether to set `requires_grad=False` for all dendritic
                             weights so they don't train
    :param dendritic_layer_class: dendritic layer class to use for each hidden layer
    :param output_nonlinearity: nonlinearity to apply to final output layer.
                                'None' of no nonlinearity.
                    _____
                   |_____|    # classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # first linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(
        self, input_size, output_size, hidden_sizes, num_segments, dim_context,
        kw, kw_percent_on=0.05, context_percent_on=1.0, weight_sparsity=0.95, 
        output_nonlinearity=None, dendritic_layer_class=AbsoluteMaxGatingDendriticLayer
    ):

        assert kw_percent_on is None or (kw_percent_on >= 0.0 and kw_percent_on < 1.0)
        assert context_percent_on >= 0.0

        if kw_percent_on == 0.0:
            kw = False

        super().__init__()

        if num_segments == 1:
            # use optimized 1 segment class
            dendritic_layer_class = OneSegmentDendriticLayer

        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw
        self.kw_percent_on = kw_percent_on
        self.weight_sparsity = weight_sparsity
        self.dendrite_weight_sparsity = dendrite_weight_sparsity
        self.output_nonlinearity = output_nonlinearity

        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()

        dendrite_sparsity = self.dendrite_weight_sparsity

        # Allow user to specify multiple layer types, with backward compatibility.
        # Just specify dendritic_layer_class as a module, and automatically broadcast
        # to a list of modules. Or, specify a list of customized modules.
        if not isinstance(dendritic_layer_class, list):
            dendritic_layer_classes = [dendritic_layer_class
                                       for i in
                                       range(len(self.hidden_sizes))]
        else:
            dendritic_layer_classes = dendritic_layer_class
        for i in range(len(self.hidden_sizes)):
            curr_dend = dendritic_layer_classes[i](
                module=nn.Linear(input_size, self.hidden_sizes[i], bias=True),
                num_segments=num_segments,
                dim_context=dim_context,
                module_sparsity=self.weight_sparsity,
                dendrite_sparsity=dendrite_sparsity,
            )

            if freeze_dendrites:
                # Dendritic weights will not be updated during backward pass
                for name, param in curr_dend.named_parameters():
                    if "segments" in name:
                        param.requires_grad = False

            if self.kw:
                curr_activation = KWinners(n=hidden_sizes[i],
                                           percent_on=kw_percent_on,
                                           k_inference_factor=1.0,
                                           boost_strength=0.0,
                                           boost_strength_factor=0.0)
            else:
                curr_activation = nn.ReLU()

            self._layers.append(curr_dend)
            self._activations.append(curr_activation)

            input_size = self.hidden_sizes[i]

        self._single_output_head = not isinstance(output_size, Iterable)
        if self._single_output_head:
            output_size = (output_size,)

        self._output_layers = nn.ModuleList()
        for out_size in output_size:
            output_layer = nn.Sequential()
            output_linear = SparseWeights(module=nn.Linear(input_size, out_size),
                                          sparsity=weight_sparsity, allow_extremes=True)
            if weight_init == "modified":
                self._init_sparse_weights(
                    output_linear, 1 - kw_percent_on if kw else 0.0)
            output_layer.add_module("output_linear", output_linear)

            if self.output_nonlinearity is not None:
                output_layer.add_module("non_linearity", output_nonlinearity)
            self._output_layers.append(output_layer)

    def forward(self, x, context=None):
        assert (context is not None) or (self.num_segments == 0)
        for layer, activation in zip(self._layers, self._activations):
            x = activation(layer(x, context))

        if self._single_output_head:
            return self._output_layers[0](x)
        else:
            return [out_layer(x) for out_layer in self._output_layers]

    # ------ Weight initialization functions ------
    @staticmethod
    def _init_sparse_weights(m, input_sparsity):
        """
        Modified Kaiming weight initialization that considers input sparsity and weight
        sparsity.
        """
        input_density = 1.0 - input_sparsity
        weight_density = 1.0 - m.sparsity
        _, fan_in = m.module.weight.size()
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.module.weight, -bound, bound)
        m.apply(rezero_weights)

    @staticmethod
    def _init_sparse_dendrites(m, input_sparsity):
        """
        Modified Kaiming initialization for dendrites segments that consider input
        sparsity and dendritic weight sparsity.
        """
        # Assume `m` is an instance of `DendriticLayerBase`
        if m.segments is not None:
            input_density = 1.0 - input_sparsity
            weight_density = 1.0 - m.segments.sparsity
            fan_in = m.dim_context
            bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
            nn.init.uniform_(m.segment_weights, -bound, bound)
            m.apply(rezero_weights)    
