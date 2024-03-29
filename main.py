import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config_dict
from pretrain import pretrain
import sys
from resnet18 import ResNet18
from heuristic_trainer import HeuristicTrainer
from iterative_trainer import IterativeTrainer
from bilevel_trainer import BilevelTrainer
from eval_utils import tst_accuracy

# Configure GPU or CPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper params
batch_size = config_dict['batch_size']
epochs = config_dict['epochs']
intl_prune_rate = config_dict['initial_prune_rate']
one_time_prune_rate = config_dict['one_time_prune_rate']
filter_prune_rate = config_dict['filter_prune_rate']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def init():
    # Parsing Command Line Parameters
    if len(sys.argv) < 2:
        print("Please input the distribution for the latent variables as argv[1]："
              "1. Heuristics-based pruning; 2. Interative Magnitude Pruning; "
              "3. Fine-grained Bi-level Pruning; 4. Filter-wise Bi-level Pruning")
        return

    model_param = int(sys.argv[1])
    model_path = ''

    if model_param == 3:
        model = ResNet18(structured_flag=True).to(device)
        pretrain_path = config_dict['pretrained_structured_file']
    else:
        model = ResNet18().to(device)
        pretrain_path = config_dict['pretrained_unstructured_file']

    if model_param == 0:
        model_path += 'heuristic'
        trainer = HeuristicTrainer(model, train_loader, device)
    elif model_param == 1:
        model_path += 'iterative'
        trainer = IterativeTrainer(model, train_loader, intl_prune_rate, device)
    elif model_param == 2:
        model_path += 'bilevel_finegrained'
        trainer = BilevelTrainer(model, train_loader, one_time_prune_rate, device)
    elif model_param == 3:
        model_path += 'bilevel_filterwise'
        trainer = BilevelTrainer(model, train_loader, filter_prune_rate, device, structured_flag=True)
    model_path += '_resnet18.pth'

    return model, trainer, pretrain_path


def train(trainer):

    model = trainer.train()

    return model


if __name__ == "__main__":

    res18, trainer, pretrain_path = init()

    # pretrain
    res18 = pretrain(res18, train_loader, device, pretrain_path)

    # test
    tst_accuracy(res18, test_loader, device)

    # train
    res18 = train(trainer)

    # test
    tst_accuracy(res18, test_loader, device)
