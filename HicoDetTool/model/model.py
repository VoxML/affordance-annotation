from transformers import AutoModelForImageClassification, AutoFeatureExtractor, \
    AutoModelForSequenceClassification, AutoTokenizer, AutoModel, ViTForImageClassification, AutoConfig
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
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


class MergedModel(nn.Module):
    def __init__(self, image_transformer, bert_transformer, num_classes):
        super().__init__()
        self.num_labels = num_classes
        print("Load Image Model")
        self.vit = AutoModel.from_pretrained(image_transformer, add_pooling_layer=False)

        #config = AutoConfig.from_pretrained(image_transformer)
        #self.vit = AutoModel.from_config(config)

        print("Load Bert Model")
        self.bert = AutoModel.from_pretrained(bert_transformer, add_pooling_layer=True)
        #config = AutoConfig.from_pretrained(bert_transformer)
        #self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(768 + 768, num_classes)


    def forward(self,
                pixel_values=None, labels=None, #output_attentions=None, output_hidden_states=None,
                input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        print("===========================")
        vit_outputs = self.vit(pixel_values, return_dict=True)
        vit_sequence_output = vit_outputs[0]
        vit_sequence_output = vit_sequence_output[:, 0, :]

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_bert_output = bert_outputs[1]
        print(vit_sequence_output)
        print(pooled_bert_output)

        pooled_bert_output = self.dropout(pooled_bert_output)
        print(vit_sequence_output.size())
        print(pooled_bert_output.size())
        concat = torch.cat((vit_sequence_output, pooled_bert_output), dim=1)
        print(concat.size())
        logits = self.classifier(concat)

        return SequenceClassifierOutput(
            logits=logits
        )


def initialize_image_bert_model(image_transformer, bert_transformer, num_classes):
    model = MergedModel(image_transformer, bert_transformer, num_classes=num_classes)

    print("Load Image Feature Extractor")
    image_feature_extractor = AutoFeatureExtractor.from_pretrained(image_transformer)

    print("Load Bert Tokanizer")
    bert_tokenizer = None
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_transformer)
    return model, image_feature_extractor, bert_tokenizer
