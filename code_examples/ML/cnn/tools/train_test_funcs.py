import torch
import torch.nn.functional as F

import os
import datetime as dt
import numpy as np
import pandas as pd
from PIL import Image
import wandb


def test(model, loader, device):
    test_loss, test_acc = 0.0, 0.0
    model.eval()

    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            loss = F.cross_entropy(pred, target)

            test_loss += loss.item() * target.shape[0]
            test_acc += torch.sum(torch.argmax(pred, dim=1) == target).item()

    n_samples = len(loader.dataset)
    return test_loss / n_samples, test_acc / n_samples


def train_epoch(model, optimizer, train_loader, device):
    train_loss, train_acc = 0.0, 0.0
    model.train()

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * target.shape[0]
        train_acc += torch.sum(torch.argmax(pred, dim=1) == target).item()

    n_samples = len(train_loader.dataset)
    return train_loss / n_samples, train_acc / n_samples


def train_with_wandb(model, optimizer, n_epochs, train_loader, val_loader, device,
                     scheduler=None, verbose=False, wandb_init_data=None):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    if wandb_init_data is None:
        wandb_init_data = {
            "project": "intro-to-dl-bhw-01",
            "name": "train run " + str(dt.datetime.now()),
            "config": {
                "dataset": "bhw",
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "initial_lr": optimizer.state_dict()["param_groups"][0]["lr"],
                "num_epochs": n_epochs,
                "train_loader_batch_size": train_loader.batch_size
            }
        }

    with wandb.init(**wandb_init_data) as run:
        start = dt.datetime.now()
        
        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
            val_loss, val_acc = test(model, val_loader, device)

            wandb.log({"loss/train": train_loss, "acc/train": train_acc,
                       "loss/val": val_loss, "acc/val": val_acc})

            train_loss_log.append(train_loss)
            train_acc_log.append(train_acc)
            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)

            if verbose:
                print(f"Epoch {epoch}")
                print(f" train loss: {train_loss}, train acc: {train_acc}")
                print(f" val loss: {val_loss}, val acc: {val_acc}\n")

            if scheduler is not None:
                scheduler.step()
            
#             if epoch % 20 == 0:
#                 torch.save({
#                     'epochs': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
#                 }, 'bignet_' + str(epoch) + '_epochs.pt')
        
        end = dt.datetime.now()
        total_seconds = (end - start).total_seconds()
        run.summary['total_fit_time'] = {
            'in_minutes': total_seconds / 60,
            'in_hours': total_seconds / 60 / 60
        }
        run.summary['final_val_accuracy'] = test(model, val_loader, device)[1]

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log


def make_predictions(root, transform, model, device, out_filename):
    res = pd.DataFrame(columns=['Id', 'Label'])

    for image_name in sorted(os.listdir(root)):
        image = Image.open(os.path.join(root, image_name)).convert('RGB')
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)
        pred_logits = model(image)
        pred_label = torch.argmax(pred_logits).item()
        res.loc[len(res)] = [image_name, pred_label]

    res.set_index('Id').to_csv(out_filename)
