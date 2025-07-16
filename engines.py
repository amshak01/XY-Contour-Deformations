import torch
from livelossplot import PlotLosses


def train(model, device, loader, optimizer, epochs, scheduler=None):
    """Trains a neural network model on a given dataset

    Args:
        model (nn): torch neural network to train
        device (device): device on which to train (cuda or cpu)
        loader (DataLoader): dataloader that feeds samples during training
        epochs (int): number of epochs to train for
        learning_rate (float): learning rate

    Returns:
        list[float]: list containing loss at each epoch for plotting/diagnostic purposes
    """

    liveloss = PlotLosses()
    logs = {}

    model.train()

    for i in range(epochs):
        total_loss = 0

        for batch, inputs in enumerate(loader):

            inputs = tuple(input.to(device) for input in inputs)

            optimizer.zero_grad()

            reQ = model(*inputs).real
            loss = torch.mean(reQ**2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(loader)
        logs["train"] = avg_loss

        liveloss.update(logs)
        liveloss.send()