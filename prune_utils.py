import torch
import torch.nn as nn


# based on abs of weights
def heuristic_pruning(resnet18_instance, threshold):
    total_weights = 0
    total_pruned = 0

    for name, module in resnet18_instance.model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights = torch.abs(module.weight.data)
            mask = weights > threshold
            module.activate_flag.data = mask.float()

            # calculate accuracy
            num_weights = mask.numel()
            num_pruned = num_weights - mask.sum().item()
            pruned_perc = num_pruned / num_weights * 100
            print(f'Layer {name}: {pruned_perc:.2f}% pruned')

            total_weights += num_weights
            total_pruned += num_pruned

    total_pruned_perc = total_pruned / total_weights * 100
    print(f'Total model pruning: {total_pruned_perc:.2f}%')
    return resnet18_instance


# pruning progressively
def iterative_pruning(resnet18_instance, prune_rate):
    for name, module in resnet18_instance.model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights = torch.abs(module.weight.data)
            # already pruned in previous iterations
            already_pruned = (module.activate_flag.data == 0)
            weights[already_pruned] = 0

            threshold = torch.quantile(weights, prune_rate)
            mask = weights > threshold
            module.activate_flag.data = mask.float()

    return resnet18_instance


# based on rate
def onetime_pruning_by_rate(resnet18_instance, prune_rate):
    for name, module in resnet18_instance.model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights = torch.abs(module.weight.data)
            threshold = torch.quantile(weights, prune_rate)
            mask = weights > threshold
            module.activate_flag.data = mask.float()
    return resnet18_instance


# filter-wise pruning, using norms
def filter_pruning_by_rate(resnet18_instance, prune_rate, p=1):
    all_norms = []

    for name, module in resnet18_instance.model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            norms = torch.norm(module.weight.data.view(module.out_channels, -1), p=p, dim=1)
            all_norms.append(norms)

    # order
    all_norms = torch.cat(all_norms)
    threshold = torch.quantile(all_norms, prune_rate)

    # prune by rate
    for name, module in resnet18_instance.model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            norms = torch.norm(module.weight.data.view(module.out_channels, -1), p=p, dim=1)
            mask = norms > threshold
            reshape_size = (module.out_channels,) + (1,) * (module.weight.data.dim() - 1)
            module.activate_flag.data = mask.view(reshape_size).float()

    return resnet18_instance


# freeze vars by name
def freeze_vars_by_key(resnet18_instance, key):
    for i, v in resnet18_instance.model.named_modules():
        if hasattr(v, key):
            if getattr(v, key) is not None:
                getattr(v, key).requires_grad = False
    return resnet18_instance


# unfreeze vars by name
def unfreeze_vars_by_key(resnet18_instance, key):
    for i, v in resnet18_instance.model.named_modules():
        if hasattr(v, key):
            if getattr(v, key) is not None:
                getattr(v, key).requires_grad = True
    return resnet18_instance
