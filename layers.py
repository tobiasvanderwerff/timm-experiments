import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional batch normalization. Adds a learned bias to the batchnorm affine
    parameters.

    Standard batchnorm:

        y = weight * norm(x) + bias

    Conditional batchnorm:
    
        y = (weight + wd) * norm(x) + (bias + bd)

    where `wd` and `bd` are learned parameters that are added to the original
    batchnorm weights and biases. These learned bias and weight vectors are of
    size C, where C is the number of output channels.
    """

    def __init__(
        self,
        batchnorm_layer: nn.BatchNorm2d,
        adaptation_num_hidden: int = 128,
    ):
        super().__init__()
        self.bn = batchnorm_layer

        # self.adapt = nn.Sequential(  # 1-hidden-layer MLP
        #     nn.Linear(writer_code_size, adaptation_num_hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(adaptation_num_hidden, 2 * batchnorm_layer.num_features),
        # )
        # self.adapt = nn.Linear(writer_code_size, 2 * batchnorm_layer.num_features)

        # Define the learnable affine parameters.
        self.weight = nn.Parameter(torch.zeros(batchnorm_layer.num_features))
        self.bias = nn.Parameter(torch.zeros(batchnorm_layer.num_features))

        # Save batchnorm affine parameters.
        self.weight_org = self.bn.weight.detach().clone()
        self.bias_org = self.bn.bias.detach().clone()

        # Reset affine parameters to identify function.
        with torch.no_grad():
            self.bn.weight.fill_(1)
            self.bn.bias.fill_(0)
        self.bn.weight.requires_grad = False
        self.bn.bias.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor of shape (N, n_channels, h, w))
        """
        bsz, n_channels = x.shape[:2]
        self.weight_org, self.bias_org = self.weight_org.to(x.device), self.bias_org.to(x.device)

        x = self.bn(x)

        # weight_and_bias = self.adapt(self.writer_code)  # shape: (N, 2 * n_channels)
        # weight_delta = weight_and_bias[:, :n_channels]
        # bias_delta = weight_and_bias[:, n_channels:]

        weight = (self.weight_org + self.weight).unsqueeze(0).expand(bsz, -1).view(bsz, n_channels, 1, 1)
        bias = (self.bias_org + self.bias).unsqueeze(0).expand(bsz, -1).view(bsz, n_channels, 1, 1)

        # Print weight and bias norm
        print_weight_bias_norm = False
        if print_weight_bias_norm:
            import random
            if random.random() < 0.01:
                print(f"C={n_channels} -- Weight norm: {torch.norm(self.weight).item():.2f}, Bias norm: {torch.norm(self.bias).item():.2f}")

        return x * weight + bias

    @staticmethod
    def replace_bn2d(module: nn.Module, adaptation_num_hidden: int = 128):
        """
        Replace all nn.BatchNorm2d layers in a module with ConditionalBatchNorm2d layers.

        Returns:
            list of all newly added ConditionalBatchNorm2d modules
        """
        new_mods = []
        if isinstance(module, ConditionalBatchNorm2d):
            return new_mods
        if isinstance(module, nn.Sequential):
            for i, m in enumerate(module):
                if type(m) == nn.BatchNorm2d:
                    new_bn = ConditionalBatchNorm2d(m, adaptation_num_hidden)
                    module[i] = new_bn
                    new_mods.append(new_bn)
        else:
            for attr_str in dir(module):
                attr = getattr(module, attr_str)
                if type(attr) == nn.BatchNorm2d:
                    new_bn = ConditionalBatchNorm2d(attr, adaptation_num_hidden)
                    setattr(module, attr_str, new_bn)
                    new_mods.append(new_bn)

        for child_module in module.children():
            new_mods.extend(
                ConditionalBatchNorm2d.replace_bn2d(
                    child_module, adaptation_num_hidden
                )
            )
        return new_mods
