import torch
import torch.nn as nn
import torch.optim as optim
from config import config_dict

epochs = config_dict['pretrain_epochs']
pretrain_lr = config_dict['pretrain_lr']


def pretrain(model, pretrain_loader, model_path='./pretrained_resnet18.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check if pretrained model exists
    try:
        model.load_state_dict(torch.load(model_path))
        print("Loaded pretrained model from", model_path)
    except FileNotFoundError:
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
