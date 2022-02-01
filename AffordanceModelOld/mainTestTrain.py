import configparser
import torch
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, trainer_utils
from transformers import default_data_collator
from transformers.integrations import TensorBoardCallback
from datasets import load_metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models.datasets import load_test_data, HicoDetDataset
from models.models import ViTForImageClassification2

metric = load_metric("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


def train_and_eval(train: Dataset, dev: Dataset, config):
    model = ViTForImageClassification2()
    model.train()
    data_collator = default_data_collator

    args = TrainingArguments(
        config.get("MODEL", "name"),
        evaluation_strategy=trainer_utils.IntervalStrategy.EPOCH,
        save_strategy=trainer_utils.IntervalStrategy.EPOCH,
        learning_rate=config.getfloat("MODEL", "learning_rate"),
        per_device_train_batch_size=config.getint("MODEL", "train_batch_size"),
        per_device_eval_batch_size=config.getint("MODEL", "eval_batch_size"),
        num_train_epochs=config.getint("MODEL", "epochs"),
        weight_decay=config.getfloat("MODEL", "learning_rate"),
        #load_best_model_at_end=True,
        #metric_for_best_model="f1",
        logging_dir='logs',
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback()]
    )
    trainer.train()
    print(".....")
    #trainer.evaluate()
    return trainer


if __name__ == '__main__':
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    data, data_label = load_test_data(configp)
    count_label = Counter(data_label)
    print(count_label)

    train_dataset = HicoDetDataset(data[:2750], data_label[:2750])
    #dev_dataset = HicoDetDataset(data[2750:3000], data_label[2750:3000])
    #test_dataset = HicoDetDataset(data[3000:], data_label[3000:])
    test_dataset = HicoDetDataset(data[2750:], data_label[2750:])

    #trained_model = train_and_eval(train_dataset, dev_dataset, configp)


    trained_model = ViTForImageClassification2()
    trained_model.load_state_dict(torch.load("test3/checkpoint-2750/pytorch_model.bin"))
    trained_model.eval()

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=configp.getint("MODEL", "eval_batch_size"),
                                                 shuffle=False, num_workers=1)
    all_predictions = []
    all_gold_labels = []
    for batch in testloader:
        with torch.no_grad():
            pred_test = trained_model(**batch)
        predictions = np.argmax(pred_test.logits, axis=1)
        all_predictions.extend(predictions.tolist())
        all_gold_labels.extend(batch["labels"].tolist())
    print(all_predictions)
    print(all_gold_labels)
    f1_micro = metric.compute(predictions=all_predictions, references=all_gold_labels, average="micro")
    f1_macro = metric.compute(predictions=all_predictions, references=all_gold_labels, average="macro")
    conf_matrix = confusion_matrix(all_gold_labels, all_predictions)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["-", "O", "T"])
    disp.plot()
    plt.title("F1_Micro: " + str(f1_micro) + " ; F1_Macro" + str(f1_macro))
    plt.show()
    print(conf_matrix)

