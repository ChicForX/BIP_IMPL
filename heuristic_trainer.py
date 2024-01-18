import torch
from prune_utils import heuristic_pruning, freeze_vars_by_key
from config import config_dict
import torch.nn.functional as F


class HeuristicTrainer:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['heuristic_lr'])
        self.epochs = config_dict['heuristic_epochs']

    def train(self):
        self.model = heuristic_pruning(self.model, config_dict['heuristic_threshold'])
        self.model = freeze_vars_by_key(self.model, config_dict['activate_key'])

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}')

        return self.model


