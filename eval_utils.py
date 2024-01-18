import torch


def tst_accuracy(model, tst_loader, device):
    correct = 0.0
    total = 0.0
    for data, target in tst_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        predicted = torch.max(output.data, 1)
        total += predicted.size(0)
        correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy
