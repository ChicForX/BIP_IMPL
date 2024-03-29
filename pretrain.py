import torch
import torch.nn as nn
import torch.optim as optim
from config import config_dict
import os

epochs = config_dict['pretrain_epochs']
pretrain_lr = config_dict['pretrain_lr']


def pretrain(model, pretrain_loader, device, model_path):

    # check if pretrained model exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained model from", model_path)
    else:
        print("No pretrained model found, starting training.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=pretrain_lr, momentum=0.9)

        # pretrain
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(pretrain_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(pretrain_loader)}')

        # save model
        torch.save(model.state_dict(), model_path)
        print('Finished Training and saved model to', model_path)
    return model
