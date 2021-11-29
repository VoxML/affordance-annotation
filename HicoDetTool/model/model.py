from transformers import AutoModelForImageClassification, AutoFeatureExtractor, \
    AutoModelForSequenceClassification, AutoTokenizer
from torchvision import datasets, models, transforms
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "efficient":
        # TODO
        pass
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def initialize_transformer_models(model_name, num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    return model_ft, feature_extractor

class PoseModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, dropout, num_classes):
        super(PoseModel, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size2 = hidden_size2
        print(self.input_size)
        print(hidden_size)
        self.l1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop_layer = nn.Dropout(p=dropout)
        if self.hidden_size2 > 0:
            self.l2 = nn.Linear(hidden_size, hidden_size2)
            self.l3 = nn.Linear(hidden_size2, num_classes)
        else:
            self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.drop_layer(out)
        if self.hidden_size2 > 0:
            out = self.l2(out)
            out = self.relu(out)
            out = self.drop_layer(out)
        out = self.l3(out)
        return out


def initialize_pose_model(input_size, hidden_size, hidden_size2, dropout, num_classes):
    model_ft = PoseModel(input_size, hidden_size, hidden_size2, dropout, num_classes)
    return model_ft


def initialize_bert_model(model_name, num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(model_ft)
    exit()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    tokenizer = AutoTokenizer.from_pretrained(model_name, num_classes=num_classes)

    return model_ft, tokenizer