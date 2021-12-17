"""
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
from __future__ import print_function
from __future__ import division
from model.model import initialize_pose_model
from model.dataset import HicoDetAffordancePoseDataset
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm


def train_model(model, dataloaders, criterion, optimizer, with_kp_score, num_epochs=25, device=torch.device("cpu"), run=None):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        epoch_gold = []
        epoch_pred = []

        for data in tqdm(dataloaders["train"]):
            j_key = data["j_key"]
            kp_score = data["kp_score"]
            labels = data["label"]
            j_key = torch.flatten(j_key, 1)
            kp_score = torch.flatten(kp_score, 1)
            if with_kp_score:
                inputs = torch.cat((j_key, kp_score), 1)
            else:
                inputs = j_key

            inputs = inputs.to(device).double()
            labels = labels.to(device).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs.double())
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_gold.extend(labels.cpu().detach().numpy())
            epoch_pred.extend(preds.cpu().detach().numpy())

        train_loss = running_loss / len(dataloaders["train"].dataset)
        train_acc = running_corrects.double() / len(dataloaders["train"].dataset)

        train_prec_micro = precision_score(epoch_gold, epoch_pred, average='micro')
        train_prec_macro = precision_score(epoch_gold, epoch_pred, average='macro')
        train_recall_micro = recall_score(epoch_gold, epoch_pred, average='micro')
        train_recall_macro = recall_score(epoch_gold, epoch_pred, average='macro')
        train_f1_micro = f1_score(epoch_gold, epoch_pred, average='micro')
        train_f1_macro = f1_score(epoch_gold, epoch_pred, average='macro')
        train_f1_weighted = f1_score(epoch_gold, epoch_pred, average='weighted')


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        epoch_gold = []
        epoch_pred = []
        # Each epoch has a training and validation phase

        for data in tqdm(dataloaders["val"]):
            j_key = data["j_key"]
            kp_score = data["kp_score"]
            labels = data["label"]
            j_key = torch.flatten(j_key, 1)
            kp_score = torch.flatten(kp_score, 1)
            if with_kp_score:
                inputs = torch.cat((j_key, kp_score), 1)
            else:
                inputs = j_key

            inputs = inputs.to(device).double()
            labels = labels.to(device).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs.double())
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_gold.extend(labels.cpu().detach().numpy())
            epoch_pred.extend(preds.cpu().detach().numpy())

        eval_loss = running_loss / len(dataloaders["val"].dataset)
        eval_acc = running_corrects.double() / len(dataloaders["val"].dataset)

        eval_prec_micro = precision_score(epoch_gold, epoch_pred, average='micro')
        eval_prec_macro = precision_score(epoch_gold, epoch_pred, average='macro')
        eval_recall_micro = recall_score(epoch_gold, epoch_pred, average='micro')
        eval_recall_macro = recall_score(epoch_gold, epoch_pred, average='macro')
        eval_f1_micro = f1_score(epoch_gold, epoch_pred, average='micro')
        eval_f1_macro = f1_score(epoch_gold, epoch_pred, average='macro')
        eval_f1_weighted = f1_score(epoch_gold, epoch_pred, average='weighted')

        print('TrainLoss: {:.4f} TrainAcc: {:.4f}; TestLoss: {:.4f} TestAcc: {:.4f}'.format(train_loss, train_acc, eval_loss, eval_acc))

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if run:
            run.log({"epoch": epoch,
                   "train_loss": train_loss, "train_acc": train_acc,
                    "train_prec_micro": train_prec_micro, "train_prec_macro": train_prec_macro,
                    "train_recall_micro": train_recall_micro, "train_recall_macro": train_recall_macro,
                    "train_f1_micro": train_f1_micro, "train_f1_macro": train_f1_macro, "train_f1_weighted": train_f1_weighted,
                    "eval_loss": eval_loss, "eval_acc": eval_acc,
                    "eval_prec_micro": eval_prec_micro, "eval_prec_macro": eval_prec_macro,
                    "eval_recall_micro": eval_recall_micro, "eval_recall_macro": eval_recall_macro,
                    "eval_f1_micro": eval_f1_micro, "eval_f1_macro": eval_f1_macro, "eval_f1_weighted": eval_f1_weighted,
                    "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=epoch_gold, preds=epoch_pred, class_names=["-", "G", "T"])})

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    batch_size = configp.getint("MODEL", "batch_size")
    num_epochs = configp.getint("MODEL", "num_epochs")
    data_dir = configp.get("HICODET", "hico_images")
    num_classes = 3
    # lr = 0.001
    # optimizer = "sgd"
    for pose_type in ["halpe136", "halpe26", "fast-pose-duc"]:
        for lr in [0.01, 0.001, 0.0001]:
            for optimizer in ["sgd", "adam", "adamw", "adagrad"]:
                for hidden_size in [25, 50, 100, 200, 300]:
                    for hidden_size2 in [0, 25, 100, 200]:
                        for dropout in [0.0, 0.4, 0.5, 0.6]:
                            for filter_kp_score in [True, False]:
                                for with_kp_score in [True, False]:
                                    for scale in [True, False]:
                                        if hidden_size2 > hidden_size:
                                            continue

                                        if os.path.exists(
                                                f'{pose_type}_{optimizer}_{lr}_{hidden_size}_{hidden_size2}_{dropout}_{filter_kp_score}_{with_kp_score}_{scale}_weights.pth'):
                                            continue
                                        run = wandb.init(project="affordance_image_pose", config={"model_name": pose_type, "lr": lr, "opimizer": optimizer,
                                                                                       "hidden_size": hidden_size, "hidden_size2": hidden_size2, "dropout": dropout,
                                                                                        "filter_kp_score": filter_kp_score,
                                                                                         "with_kp_score": with_kp_score, "scale":scale}, reinit=True)

                                        print("Initializing Datasets and Dataloaders...")

                                        configp["POSE"]["train_dataset"] = "data/poses/" + pose_type + "-train.json"
                                        configp["POSE"]["test_dataset"] = "data/poses/" + pose_type + "-test.json"
                                        # Create training and validation datasets
                                        image_datasets = {"train": HicoDetAffordancePoseDataset(config=configp, filter_kp_score=filter_kp_score, scale=scale, train=True),
                                                          "val": HicoDetAffordancePoseDataset(config=configp, filter_kp_score=filter_kp_score, scale=scale, train=False)}
                                        # Create training and validation dataloaders

                                        dataloaders_dict = {
                                            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
                                            ['train', 'val']}

                                        input_size = image_datasets["train"].data_size
                                        if not with_kp_score:
                                            input_size = input_size * 2 / 3
                                        model_ft = initialize_pose_model(input_size, hidden_size,hidden_size2, dropout, num_classes)
                                        model_ft = model_ft.double()
                                        #run.watch(model_ft, log_freq=100)


                                        # Detect if we have a GPU available
                                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                                        # Send the model to GPU
                                        model_ft = model_ft.to(device)

                                        params_to_update = model_ft.parameters()

                                        # Observe that all parameters are being optimized
                                        if optimizer == "sgd":
                                            optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)
                                        elif optimizer == "adam":
                                            optimizer_ft = optim.Adam(params_to_update, lr=lr)
                                        elif optimizer == "adamw":
                                            optimizer_ft = optim.AdamW(params_to_update, lr=lr)
                                        elif optimizer == "adagrad":
                                            optimizer_ft = optim.Adagrad(params_to_update, lr=lr)
                                        else:
                                            print("could not find:", optimizer)

                                        criterion = nn.CrossEntropyLoss()

                                        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,with_kp_score=with_kp_score, num_epochs=num_epochs, device=device, run=run)

                                        torch.save(model_ft.state_dict(), f'{pose_type}_{optimizer}_{lr}_{hidden_size}_{hidden_size2}_{dropout}_{filter_kp_score}_{with_kp_score}_{scale}_weights.pth')
                                        torch.save(model_ft, f'{pose_type}_{optimizer}_{lr}_{hidden_size}_{hidden_size2}_{dropout}_{filter_kp_score}_{with_kp_score}_{scale}_weights.pth')
                                        print(hist)
                                        run.finish()