import torch.optim as optim
import torch.nn.functional as F
from config import config_dict
from prune_utils import iterative_pruning, freeze_vars_by_key


class IterativeTrainer:
    def __init__(self, model, train_loader, prune_rate, device, momentum=0.9):
        self.model = model
        self.train_loader = train_loader
        self.prune_rate = prune_rate
        self.epochs = config_dict['itr_epochs']
        self.optimizer = optim.SGD(model.parameters(), lr=config_dict['itr_lr'], momentum=momentum)
        self.device = device

    def train(self):
        # prune manually in iterative_pruning()
        self.model = freeze_vars_by_key(self.model, config_dict['activate_key'])

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            # prune every 10 epochs
            if epoch % 10 == 0:
                self.prune_rate *= 0.8
                self.model = iterative_pruning(self.model, prune_rate=self.prune_rate)

            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                # retrain
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}')

        return self.model
