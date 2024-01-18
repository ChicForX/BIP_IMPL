import torch.optim as optim
import torch.nn.functional as F
from config import config_dict
from prune_utils import freeze_vars_by_key, unfreeze_vars_by_key, onetime_pruning_by_rate


class BilevelTrainer:
    def __init__(self, model, train_loader, prune_rate, momentum=0.9):
        self.model = model
        self.train_loader = train_loader
        self.prune_rate = prune_rate
        self.epochs = config_dict['itr_epochs']
        self.optimizer_pr = optim.SGD(model.parameters(), lr=config_dict['pr_lr'], momentum=momentum)
        self.optimizer_ft = optim.SGD(model.parameters(), lr=config_dict['ft_label'], momentum=momentum)
        self.epoch_label = config_dict['ft_label']

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            # iteratively prune & retrain in each epoch
            self.epoch_label += 1
            self.epoch_label = self.epoch_label % 2

            for data, target in self.train_loader:

                if self.epoch_label == config_dict['ft_label']:
                    #  ----------------- pruning ---------------- #
                    self.model = freeze_vars_by_key(self.model, config_dict['weight_key'])
                    self.model = freeze_vars_by_key(self.model, config_dict['bias_key'])
                    self.model = unfreeze_vars_by_key(self.model, config_dict['activate_key'])
                    self.optimizer_pr.zero_grad()
                    self.model = onetime_pruning_by_rate(self.model, self.prune_rate)
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.optimizer_pr.step()

                else:
                    #  --------------- fine-tuning -------------- #
                    self.model = unfreeze_vars_by_key(self.model, config_dict['weight_key'])
                    self.model = unfreeze_vars_by_key(self.model, config_dict['bias_key'])
                    self.model = freeze_vars_by_key(self.model, config_dict['activate_key'])
                    self.optimizer_ft.zero_grad()
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.optimizer_ft.step()

                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}')

        return self.model
