import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, structured_flag=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18()
        self.structured_flag = structured_flag

        for name, module in self.model.named_modules():
            if self.structured_flag and isinstance(module, nn.Conv2d):
                # Filter-wise pruning for Conv2d
                module.register_buffer('activate_flag', torch.ones(module.out_channels, dtype=torch.bool))
            elif not self.structured_flag and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                # Fine-grained pruning for Conv2d and Linear
                module.register_buffer('activate_flag', torch.ones_like(module.weight.data))

    def forward(self, x):
        for name, module in self.model.named_children():
            if isinstance(module, nn.Conv2d):
                if self.structured_flag:
                    # filter-wise
                    reshape_size = (module.out_channels,) + (1,) * (module.weight.dim() - 1)
                    prune_mask = module.activate_flag.view(reshape_size)
                    temp_weight = module.weight.data * prune_mask
                else:
                    # fine-grained
                    temp_weight = module.weight.data * module.activate_flag

                # forward using temp weight
                x = F.conv2d(x, temp_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
            elif isinstance(module, nn.Linear):
                x = torch.flatten(x, 1)
                if self.structured_flag:
                    x = module(x)
                else:
                    # fine-grained
                    temp_weight = module.weight.data * module.activate_flag

                    # forward using temp weight
                    x = F.linear(x, temp_weight, module.bias)
            else:
                # normally forward
                x = module(x)

        return x
