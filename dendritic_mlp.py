import abc
import math
from itertools import product
from collections.abc import Iterable
from collections import namedtuple
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import init
from k_winners import KWinners
from sparse_weights import SparseWeights, SparseWeights2d, 
                           rezero_weights, HasRezeroWeights
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

class DendriteSegments(torch.nn.Module, HasRezeroWeights):
    """
    This implements dendrite segments over a set of units. Each unit has a set of
    segments modeled by a linear transformation from a context vector to output value
    for each segment.
    """

    def __init__(self, num_units, num_segments, dim_context, sparsity, bias=None):
        """
        :param num_units: number of units i.e. neurons;
                          each unit will have it's own set of dendrite segments
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param num_segments: number of dendrite segments per unit
        :param sparsity: sparsity of connections;
                        this is over each linear transformation from
                        dim_context to num_segments
        """
        super().__init__()

        # Save params.
        self.num_units = num_units
        self.num_segments = num_segments
        self.dim_context = dim_context
        self.sparsity = sparsity

        # TODO: Use named dimensions.
        weights = torch.Tensor(num_units, num_segments, dim_context)
        self.weights = torch.nn.Parameter(weights)

        # Create a bias per unit per segment.
        if bias:
            biases = torch.Tensor(num_units, num_segments)
            self.biases = torch.nn.Parameter(biases)
        else:
            self.register_parameter("biases", None)
        self.reset_parameters()

        # Create a random mask per unit per segment (dims=[0, 1])
        zero_mask = random_mask(
            self.weights.shape,
            sparsity=sparsity,
            dims=[0, 1]
        )

        # Use float16 because pytorch distributed nccl doesn't support bools.
        self.register_buffer("zero_mask", zero_mask.half())

        self.rezero_weights()

    def extra_repr(self):
        return (
            f"num_units={self.num_units}, "
            f"num_segments={self.num_segments}, "
            f"dim_context={self.dim_context}, "
            f"sparsity={self.sparsity}, "
            f"bias={self.biases is not None}"
        )

    def reset_parameters(self):
        """Initialize the linear transformation for each unit."""
        for unit in range(self.num_units):
            weight = self.weights[unit, ...]
            if self.biases is not None:
                bias = self.biases[unit, ...]
            else:
                bias = None
            init_linear_(weight, bias)

    def rezero_weights(self):
        self.weights.data.masked_fill_(self.zero_mask.bool(), 0)

    def forward(self, context):
        """
        Matrix-multiply the context with the weight tensor for each dendrite segment.
        This is done for each unit and so the output is of length num_units.
        """

        # Matrix multiply using einsum:
        #    * b => the batch dimension
        #    * k => the context dimension; multiplication will be along this dimension
        #    * ij => the units and segment dimensions, respectively
        # W^C * M^C * C -> num_units x num_segments
        output = torch.einsum("ijk,bk->bij", self.weights, context)

        if self.biases is not None:
            output += self.biases
        return output


def init_linear_(weight, bias=None):
    """
    Performs the default initilization of a weight and bias parameter
    of a linear layaer; done in-place.
    """
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


def random_mask(size, sparsity, dims=None, **kwargs):
    """
    This creates a random off-mask (True => off) of 'size' with the specified 'sparsity'
    level along 'dims'. If 'dims' is 1, for instance, then `mask[:, d, ...]` has the
    desired sparsity for all d. If dims is a list, say [0, 1], then `mask[d1, d2, ...]`
    will have the desired sparsity level for all d1 and d2. If None, the sparsity is
    applied over the whole tensor.
    :param size: shape of tensor
    :param sparsity: fraction of non-zeros
    :param dims: which dimensions to apply the sparsity
    :type dims: int or iterable
    :param kwargs: keywords args passed to torch.ones;
                   helpful for specifying device, for instace
    """

    assert 0 <= sparsity <= 1

    # Start with all elements off.
    mask = torch.ones(size, **kwargs)

    # Find sparse submasks along dims; recursively call 'random_mask'.
    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = [dims]

        # Loop all combinations that index through dims.
        # The 1D case is equivalent to range.
        dim_lengths = [mask.shape[dim] for dim in dims]
        dim_indices = product(*[range(dl) for dl in dim_lengths])

        for idxs in dim_indices:

            # For example, this may yield a slice that gives
            # `mask[dim_slice] == mask[:, 0, 0]` where `dims=[1, 2]`.
            dim_slice = [
                idxs[dims.index(d)] if d in dims else slice(None)
                for d in range(len(mask.shape))
            ]

            # Assign the desired sparsity to the submask.
            sub_mask = mask[dim_slice]
            sub_mask[:] = random_mask(
                sub_mask.shape,
                sparsity, **kwargs, dims=None
            )

        return mask

    # Randomly choose indices to make non-zero ("nz").
    mask_flat = mask.view(-1)  # flattened view
    num_total = mask_flat.shape[0]
    num_nz = int(round((1 - sparsity) * num_total))
    on_indices = np.random.choice(num_total, num_nz, replace=False)
    mask_flat[on_indices] = False

    return mask

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

class DendriticLayer2dBase(SparseWeights2d, metaclass=abc.ABCMeta):
    """
    Base class for all 2d Dendritic Layer modules.
    Similar to the DendriticLayerBase class, the output from the dendrite segments
    is applied to the output of each channel. Thus, each channel output gets
    modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        :param module: conv2d module which performs the forward pass
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.segments = None
        super().__init__(module, sparsity=module_sparsity)

        self.segments = DendriteSegments(
            num_units=module.out_channels,
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """
        Computes the forward pass through the `torch.nn.Conv2d` module and applies the
        output of the dendrite segments.
        """
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)
    
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
    :param weight_sparsity: the sparsity level of feed-forward weights.
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
        kw, kw_percent_on=0.05, weight_sparsity=0.95, output_nonlinearity=None,
        dendritic_layer_class=AbsoluteMaxGatingDendriticLayer, context_percent_on=1.0
    ):

        assert kw_percent_on is None or (kw_percent_on >= 0.0 and kw_percent_on < 1.0)

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
        self.output_nonlinearity = output_nonlinearity

        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()

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
                dendrite_sparsity=0.,
            )
            
            # Scale weights to be sampled from the new initialization U(-h, h) where
            # h = sqrt(1 / (weight_density * previous_layer_percent_on))
            if i == 0:
                # first hidden layer can't have kw input
                self._init_sparse_weights(curr_dend, 0.0)
            else:
                self._init_sparse_weights(
                    curr_dend,
                    1 - kw_percent_on if kw else 0.0
                )

            self._init_sparse_dendrites(curr_dend, 1 - context_percent_on)
            
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
            self._init_sparse_weights(output_linear, 1 - kw_percent_on if kw else 0.0)
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
