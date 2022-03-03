import random
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from dataset import HicoDetDataset
from models import AbsoluteOrientationNet

possible_orientations = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1]]
EPOCHS = 50
LEARNING_RATE = 0.01
FILTER_OBJ = None

def train_absolute_model(train_loader, test_loader):
    model = AbsoluteOrientationNet(6, 6, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    baseline_train_label = torch.zeros(3)
    for epoch in tqdm(range(EPOCHS)):
        for data in train_loader:
            input_data = torch.cat([data["up"], data["front"]], dim=-1)
            labels = data["affordance_lab"]
            if epoch == 0:
                for x in labels:
                    baseline_train_label[x] += 1

            outputs = model(input_data.float())
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        baseline_label = torch.zeros(3)
        for data in test_loader:
            input_data = torch.cat([data["up"], data["front"]], dim=-1)
            labels = data["affordance_lab"]
            for x in labels:
                baseline_label[x] += 1

            outputs = model(input_data.float())
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        print(n_samples)
        acc = 100.0 * n_correct / n_samples
        b_train_acc = 100.0 * baseline_label[baseline_train_label.argmax()] / n_samples
        b_test_acc = 100.0 * baseline_label.max() / n_samples
        print(baseline_train_label)
        print(baseline_label)
        print(f'Accuracy of the network on the test images: {acc} %')
        print(f'BaselineTrainAccuracy of the network on the test images: {b_train_acc} %')
        print(f'BaselineAccuracy of the network on the test images: {b_test_acc} %')

        for p_up in possible_orientations:
            for p_front in possible_orientations:
                p_ori = p_up + p_front
                outputs = model(torch.tensor([p_ori])).softmax(-1)
                print(p_up, p_front, outputs[0])


if __name__ == '__main__':
    seed = 66
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data = HicoDetDataset("D:/Corpora/HICO-DET/via234_1200 items_train verified v2.json",
                          "D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015", train=True, filter_obj=FILTER_OBJ)

    test_data = HicoDetDataset("D:/Corpora/HICO-DET/via234_1200 items_train verified v2.json",
                          "D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015", train=False, filter_obj=FILTER_OBJ)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=32,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=32,
                                               shuffle=False)

    train_absolute_model(train_loader, test_loader)