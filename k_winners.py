import abc
import numpy as np
import torch
import torch.nn as nn

def update_boost_strength(m):
    """Function used to update KWinner modules boost strength. This is typically done
    during training at the beginning of each epoch.

    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(update_boost_strength)``

    :param m: KWinner module
    """
    if isinstance(m, KWinnersBase):
        m.update_boost_strength()
        
@torch.jit.script
def boost_activations(x, duty_cycles, boost_strength: float):
    """
    Boosting as documented in :meth:`kwinners` would compute
      x * torch.exp((target_density - duty_cycles) * boost_strength)
    but instead we compute
      x * torch.exp(-boost_strength * duty_cycles)
    which is equal to the former value times a positive constant, so it will
    have the same ranked order.
    :param x:
      Current activity of each unit.
    :param duty_cycles:
      The averaged duty cycle of each unit.
    :param boost_strength:
      A boost strength of 0.0 has no effect on x.
    :return:
         A tensor representing the boosted activity
    """
    if boost_strength > 0.0:
        return x.detach() * torch.exp(-boost_strength * duty_cycles)
    else:
        return x.detach()

@torch.jit.script
def kwinners(x, duty_cycles, k: int, boost_strength: float, break_ties: bool = False,
             relu: bool = False, inplace: bool = False):
    """
    A simple K-winner take all function for creating layers with sparse output.
    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.
    The boosting function is a curve defined as:
    .. math::
        boostFactors = \\exp(-boostStrength \\times (dutyCycles - targetDensity))
    Intuitively this means that units that have been active (i.e. in the top-k)
    at the target activation level have a boost factor of 1, meaning their
    activity is not boosted. Columns whose duty cycle drops too much below that
    of their neighbors are boosted depending on how infrequently they have been
    active. Unit that has been active more than the target activation level
    have a boost factor below 1, meaning their activity is suppressed and
    they are less likely to be in the top-k.
    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.
    The target activation density for each unit is k / number of units. The
    boostFactor depends on the duty_cycles via an exponential function::
            boostFactor
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> duty_cycles
                   |
              target_density
    :param x:
      Current activity of each unit, optionally batched along the 0th dimension.
    :param duty_cycles:
      The averaged duty cycle of each unit.
    :param k:
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.
    :param boost_strength:
      A boost strength of 0.0 has no effect on x.
    :param break_ties:
      Whether to use a strict k-winners. Using break_ties=False is faster but
      may occasionally result in more than k active units.
    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners
    :param inplace:
      Whether to modify x in place
    :return:
      A tensor representing the activity of x after k-winner take all.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        indices = boosted.topk(k=k, dim=1, sorted=False)[1]
        off_mask = torch.ones_like(boosted, dtype=torch.bool)
        off_mask.scatter_(1, indices, 0)

        if relu:
            off_mask.logical_or_(boosted <= 0)
    else:
        threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                     keepdim=True)[0]

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)

@torch.jit.script
def kwinners2d(x, duty_cycles, k: int, boost_strength: float, local: bool = True,
               break_ties: bool = False, relu: bool = False,
               inplace: bool = False):
    """
    A K-winner take all function for creating Conv2d layers with sparse output.
    If local=True, k-winners are chosen independently for each location. For
    Conv2d inputs (batch, channel, H, W), the top k channels are selected
    locally for each of the H X W locations. If there is a tie for the kth
    highest boosted value, there will be more than k winners.
    The boost strength is used to compute a boost factor for each unit
    represented in x. These factors are used to increase the impact of each unit
    to improve their chances of being chosen. This encourages participation of
    more columns in the learning process. See :meth:`kwinners` for more details.
    :param x:
      Current activity of each unit.
    :param duty_cycles:
      The averaged duty cycle of each unit.
    :param k:
      The activity of the top k units across the channels will be allowed to
      remain, the rest are set to zero.
    :param boost_strength:
      A boost strength of 0.0 has no effect on x.
    :param local:
      Whether or not to choose the k-winners locally (across the channels at
      each location) or globally (across the whole input and across all
      channels).
    :param break_ties:
      Whether to use a strict k-winners. Using break_ties=False is faster but
      may occasionally result in more than k active units.
    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners.
    :param inplace:
      Whether to modify x in place
    :return:
         A tensor representing the activity of x after k-winner take all.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        if local:
            indices = boosted.topk(k=k, dim=1, sorted=False)[1]
            off_mask = torch.ones_like(boosted, dtype=torch.bool)
            off_mask.scatter_(1, indices, 0)
        else:
            shape2 = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            indices = boosted.view(shape2).topk(k, dim=1, sorted=False)[1]
            off_mask = torch.ones(shape2, dtype=torch.bool, device=x.device)
            off_mask.scatter_(1, indices, 0)
            off_mask = off_mask.view(x.shape)

        if relu:
            off_mask.logical_or_(boosted <= 0)
    else:
        if local:
            threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                         keepdim=True)[0]
        else:
            threshold = boosted.view(x.shape[0], -1).kthvalue(
                x.shape[1] * x.shape[2] * x.shape[3] - k + 1, dim=1)[0]
            threshold = threshold.view(x.shape[0], 1, 1, 1)

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)

class KWinnersBase(nn.Module, metaclass=abc.ABCMeta):
    """Base KWinners class.

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting). Must be >= 0.0
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int
    """

    def __init__(
        self,
        percent_on,
        k_inference_factor=1.0,
        boost_strength=1.0,
        boost_strength_factor=1.0,
        duty_cycle_period=1000,
    ):
        super(KWinnersBase, self).__init__()
        assert boost_strength >= 0.0
        assert 0.0 <= boost_strength_factor <= 1.0
        assert 0.0 < percent_on < 1.0
        assert 0.0 < percent_on * k_inference_factor < 1.0

        self.percent_on = percent_on
        self.percent_on_inference = percent_on * k_inference_factor
        self.k_inference_factor = k_inference_factor
        self.learning_iterations = 0
        self.n = 0
        self.k = 0
        self.k_inference = 0

        # Boosting related parameters. Put boost_strength in a buffer so that it
        # is saved in the state_dict. Keep a copy that remains a Python float so
        # that its value can be accessed in 'if' statements without blocking to
        # fetch from GPU memory.
        self.register_buffer("boost_strength", torch.tensor(boost_strength,
                                                            dtype=torch.float))
        self._cached_boost_strength = boost_strength

        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._cached_boost_strength = self.boost_strength.item()

    def extra_repr(self):
        return (
            "n={0}, percent_on={1}, boost_strength={2}, boost_strength_factor={3}, "
            "k_inference_factor={4}, duty_cycle_period={5}".format(
                self.n, self.percent_on, self._cached_boost_strength,
                self.boost_strength_factor, self.k_inference_factor,
                self.duty_cycle_period
            )
        )

    @abc.abstractmethod
    def update_duty_cycle(self, x):
        r"""Updates our duty cycle estimates with the new value. Duty cycles are
        updated according to the following formula:

        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                                + newValue}{period}

        :param x:
          Current activity of each unit
        """
        raise NotImplementedError

    def update_boost_strength(self):
        """Update boost strength by multiplying by the boost strength factor.
        This is typically done during training at the beginning of each epoch.
        """
        self._cached_boost_strength *= self.boost_strength_factor
        self.boost_strength.fill_(self._cached_boost_strength)

class KWinners(KWinnersBase):
    """Applies K-Winner function to the input tensor.

    See :class:`htmresearch.frameworks.pytorch.functions.k_winners`

    :param n:
      Number of units
    :type n: int

    :param percent_on:
      The activity of the top k = percent_on * n will be allowed to remain, the
      rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param break_ties:
        Whether to use a strict k-winners. Using break_ties=False is faster but
        may occasionally result in more than k active units.
    :type break_ties: bool

    :param relu:
        This will simulate the effect of having a ReLU before the KWinners.
    :type relu: bool

    :param inplace:
       Modify the input in-place.
    :type inplace: bool
    """

    def __init__(
        self,
        n,
        percent_on,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        break_ties=False,
        relu=False,
        inplace=False,
    ):

        super(KWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.break_ties = break_ties
        self.inplace = inplace
        self.relu = relu

        self.n = n
        self.k = int(round(n * percent_on))
        self.k_inference = int(self.k * self.k_inference_factor)
        self.register_buffer("duty_cycle", torch.zeros(self.n))

    def forward(self, x):

        if self.training:
            x = kwinners(x, self.duty_cycle, self.k, self._cached_boost_strength,
                           self.break_ties, self.relu, self.inplace)
            self.update_duty_cycle(x)
        else:
            x = kwinners(x, self.duty_cycle, self.k_inference,
                           self._cached_boost_strength, self.break_ties, self.relu,
                           self.inplace)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        self.duty_cycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.duty_cycle.div_(period)

    def extra_repr(self):
        s = super().extra_repr()
        s += f", break_ties={self.break_ties}"
        if self.relu:
            s += ", relu=True"
        if self.inplace:
            s += ", inplace=True"
        return s

class KWinners2d(KWinnersBase):
    """
    Applies K-Winner function to the input tensor.

    See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`

    :param channels:
      Number of channels (filters) in the convolutional layer.
    :type channels: int

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param local:
        Whether or not to choose the k-winners locally (across the channels
        at each location) or globally (across the whole input and across
        all channels).
    :type local: bool

    :param break_ties:
        Whether to use a strict k-winners. Using break_ties=False is faster but
        may occasionally result in more than k active units.
    :type break_ties: bool

    :param relu:
        This will simulate the effect of having a ReLU before the KWinners.
    :type relu: bool

    :param inplace:
       Modify the input in-place.
    :type inplace: bool
    """

    def __init__(
        self,
        channels,
        percent_on=0.1,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        local=False,
        break_ties=False,
        relu=False,
        inplace=False,
    ):

        super(KWinners2d, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.channels = channels
        self.local = local
        self.break_ties = break_ties
        self.inplace = inplace
        self.relu = relu
        if local:
            self.k = int(round(self.channels * self.percent_on))
            self.k_inference = int(round(self.channels * self.percent_on_inference))

        self.register_buffer("duty_cycle", torch.zeros((1, channels, 1, 1)))

    def forward(self, x):

        if self.n == 0:
            self.n = np.prod(x.shape[1:])
            if not self.local:
                self.k = int(round(self.n * self.percent_on))
                self.k_inference = int(round(self.n * self.percent_on_inference))

        if self.training:
            x = kwinners2d(x, self.duty_cycle, self.k,
                             self._cached_boost_strength, self.local,
                             self.break_ties, self.relu, self.inplace)
            self.update_duty_cycle(x)
        else:
            x = kwinners2d(x, self.duty_cycle, self.k_inference,
                             self._cached_boost_strength, self.local,
                             self.break_ties, self.relu, self.inplace)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size

        scale_factor = float(x.shape[2] * x.shape[3])
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        s = x.gt(0).sum(dim=(0, 2, 3), dtype=torch.float) / scale_factor
        self.duty_cycle.reshape(-1).add_(s)
        self.duty_cycle.div_(period)

    def extra_repr(self):
        s = (f"channels={self.channels}, local={self.local}"
             f", break_ties={self.break_ties}")
        if self.relu:
            s += ", relu=True"
        if self.inplace:
            s += ", inplace=True"
        s += ", {}".format(super().extra_repr())
        return s
