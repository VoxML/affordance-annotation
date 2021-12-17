"""
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
from __future__ import print_function
from __future__ import division


from model.model import initialize_image_bert_model
from model.dataset import HicoDetAllDataset
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
import time
import copy
from tqdm import tqdm


def train_model(model, tokenizer, dataloaders, criterion, optimizer, num_epochs=25, device=torch.device("cpu"),
                run=None):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        epoch_gold = []
        epoch_pred = []

        for data in tqdm(dataloaders["train"]):
            pixel_values = data["pixel_values"].to(device)
            caption = data["caption"]
            labels = data["label"].long().to(device)

            encoded_input = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
            encoded_input = encoded_input.to(device)
            encoded_input["pixel_values"] = pixel_values
            #encoded_input["labels"] = labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(**encoded_input).logits
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * labels.size(0)
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
            pixel_values = data["pixel_values"].to(device)
            caption = data["caption"]
            labels = data["label"].long().to(device)

            encoded_input = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
            encoded_input = encoded_input.to(device)
            encoded_input["pixel_values"] = pixel_values

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(**encoded_input).logits
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * labels.size(0)
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

        print('TrainLoss: {:.4f} TrainAcc: {:.4f}; TestLoss: {:.4f} TestAcc: {:.4f}'.format(train_loss, train_acc,
                                                                            eval_loss, eval_acc))

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        exit()
        run.log({"epoch": epoch,
                 "train_loss": train_loss, "train_acc": train_acc,
                 "train_prec_micro": train_prec_micro, "train_prec_macro": train_prec_macro,
                 "train_recall_micro": train_recall_micro, "train_recall_macro": train_recall_macro,
                 "train_f1_micro": train_f1_micro, "train_f1_macro": train_f1_macro,
                 "train_f1_weighted": train_f1_weighted,
                 "eval_loss": eval_loss, "eval_acc": eval_acc,
                 "eval_prec_micro": eval_prec_micro, "eval_prec_macro": eval_prec_macro,
                 "eval_recall_micro": eval_recall_micro, "eval_recall_macro": eval_recall_macro,
                 "eval_f1_micro": eval_f1_micro, "eval_f1_macro": eval_f1_macro, "eval_f1_weighted": eval_f1_weighted,
                 "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=epoch_gold, preds=epoch_pred,
                                                         class_names=["-", "G", "T"])})

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":
    print("Load Configs")
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    base_image_model = configp.get("MODEL", "base_image_model")
    base_bert_model = configp.get("MODEL", "base_bert_model")
    batch_size = configp.getint("MODEL", "batch_size")
    num_epochs = configp.getint("MODEL", "num_epochs")
    learning_rate = configp.getfloat("MODEL", "learning_rate")
    optimizer = configp.get("MODEL", "optimizer")
    feature_extract = configp.get("MODEL", "feature_extract")

    data_dir = configp.get("HICODET", "hico_images")
    num_classes = 3

    """
        run = wandb.init(project="affordance_merged",
                     config=configp,
                     reinit=True)
    """
    print("Initialize Model")

    model_ft, image_feature_extractor, bert_tokenizer = initialize_image_bert_model(base_image_model, base_bert_model, num_classes, True, 0.1, 0.1)

    #run.watch(model_ft, log_freq=100)

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {
        "train": HicoDetAllDataset(config=configp, train=True, feature_extractor=image_feature_extractor),
        "val": HicoDetAllDataset(config=configp, train=False, feature_extractor=image_feature_extractor)}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

        # Observe that all parameters are being optimized
    if optimizer == "sgd":
        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
    elif optimizer == "adamw":
        optimizer_ft = optim.AdamW(params_to_update, lr=learning_rate)
    elif optimizer == "adagrad":
        optimizer_ft = optim.Adagrad(params_to_update, lr=learning_rate)
    else:
        print("could not find:", optimizer)
        exit()
        # optimizer_ft = optim.AdamW(params_to_update, lr=0.01)
        # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    run=None
    model_ft, hist = train_model(model_ft, bert_tokenizer, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 device=device, run=run)

    #torch.save(model_ft.state_dict(), f'{print_model_name}_{optimizer}_{lr}_{feature_extract}_weights.pth')
    #torch.save(model_ft, f'{print_model_name}_{optimizer}_{lr}_{feature_extract}_weights.pth')
    print(hist)
    #run.finish()
