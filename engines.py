import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from lattice_utils import deformed_corr_2d


def train(model, device, loader, optimizer, epochs, scheduler=None, val_loader=None, smoothing=0.01):
    """Trains a neural network model on a given dataset

    Args:
        model (nn): torch neural network to train
        device (device): device on which to train (cuda or cpu)
        loader (DataLoader): dataloader that feeds samples during training
        epochs (int): number of epochs to train for

    Returns:
        list[float]: list containing loss at each epoch for plotting/diagnostic purposes
    """
    liveloss = None
    logs = {}

    if val_loader is not None:
        liveloss = PlotLosses(groups={"loss": ["train", "val"]})
        logs = {"train": 0, "val": 0}
    else:
        liveloss = PlotLosses(groups={"loss": ["train"]})
        logs = {"train": 0}

    # liveloss = PlotLosses()
    # logs = {}

    model.train()
    # total_train_loss = 0

    for i in range(epochs):
        total_train_loss = 0

        for batch, inputs in enumerate(loader):

            inputs = tuple(input.to(device) for input in inputs)

            optimizer.zero_grad()

            loss = loss_fn(model, inputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            # # ----- temporary
            # if batch % 25 == 0 and batch > 0:
            #     logs = {'train' : total_train_loss / 25}
            #     liveloss.update(logs)
            #     liveloss.send()
            #     total_train_loss = 0

            #     if scheduler is not None:
            #         scheduler.step()
            # # -----

            if scheduler is not None:
                scheduler.step()

            if batch % 25 == 0:
                print(f"Batch {batch + 1}/{len(loader)}, Loss: {loss.item():.4f}")

        logs["train"] = total_train_loss / len(loader)

        if val_loader is not None:
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch, inputs in enumerate(val_loader):

                    inputs = tuple(input.to(device) for input in inputs)

                    loss = loss_fn(model, inputs)

                    total_val_loss += loss.item()

            logs["val"] = total_val_loss / len(val_loader)

        liveloss.update(logs)
        liveloss.send()

        model.train()


def loss_fn(model, inputs):
    """Computes the penalized loss for a model given its inputs

    Args:
        model (nn): torch neural network model
        inputs (tuple): inputs to the model
        smoothing (float): smoothing factor for the penalty term. Defaults to 0.01.
        centering (float): enforces zero mean shift field. Defaults to 0.01.

    Returns:
        Tensor: computed loss value
    """

    lats, temps, x_seps, y_seps = inputs
    shift_fields = model(temps, x_seps, y_seps).real
    reQ = deformed_corr_2d(lats, temps, shift_fields, x_seps, y_seps).real

    mean_square = torch.mean(reQ**2)

    return mean_square
