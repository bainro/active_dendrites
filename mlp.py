import numpy as np
import torch.nn as nn

class ModifiedInitMLP(nn.Module):
    """
    A standard MLP which differs only in feed-forward weight initialization: the bounds
    of the Uniform distribution used to initialization weights are
    +/- 1/sqrt(I x W x F)
    where I is the density of the input for a given layer, W is always 1.0 (since MLPs
    have dense weights), and F is fan-in. This only differs from Kaiming Uniform
    initialization by incorporating input density (I) and weight density (W). Biases
    are unaffected.
    """
  
    def __init__(self, input_size, num_classes,
                 hidden_sizes=(100, 100)):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(int(np.prod(input_size)), hidden_sizes[0]),
            nn.ReLU()
        ]
        for idx in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.classifier = nn.Sequential(*layers)
        
        # Modified Kaiming weight initialization which considers 1) the density of
        # the input vector and 2) the weight density in addition to the fan-in

        weight_density = 1.0
        input_flag = False
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                # Assume input is fully dense, but hidden layer activations are only
                # 50% dense due to ReLU
                input_density = 1.0 if not input_flag else 0.5
                input_flag = True

                _, fan_in = layer.weight.size()
                bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)

    def forward(self, x):
        return self.classifier(x)
