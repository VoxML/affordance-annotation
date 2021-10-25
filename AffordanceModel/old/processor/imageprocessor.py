from .base import ImageProcessor
from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrForSegmentation, \
    ViTFeatureExtractor, ViTForImageClassification,\
    DeiTForImageClassificationWithTeacher
from transformers import AutoFeatureExtractor


# https://huggingface.co/models?pipeline_tag=image-segmentation
class DetrImageSegmentation(ImageProcessor):

    def __init__(self, modelname='facebook/detr-resnet-50-panoptic'):
        feature_extractor = DetrFeatureExtractor.from_pretrained(modelname)
        model = DetrForSegmentation.from_pretrained(modelname)
        super().__init__(model, feature_extractor, modelname)


# https://huggingface.co/models?pipeline_tag=object-detection
class DetrObjectDetection(ImageProcessor):

    def __init__(self, modelname='facebook/detr-resnet-50'):
        feature_extractor = DetrFeatureExtractor.from_pretrained(modelname)
        model = DetrForObjectDetection.from_pretrained(modelname)
        super().__init__(model, feature_extractor, modelname)


# https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads
class VitImageClassification(ImageProcessor):

    def __init__(self, modelname='google/vit-base-patch16-224'):
        feature_extractor = ViTFeatureExtractor.from_pretrained(modelname)
        model = ViTForImageClassification.from_pretrained(modelname)
        super().__init__(model, feature_extractor, modelname)


# https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads
class DeitImageClassification(ImageProcessor):

    def __init__(self, modelname='facebook/deit-base-distilled-patch16-224'):
        feature_extractor = AutoFeatureExtractor.from_pretrained(modelname)
        model = DeiTForImageClassificationWithTeacher.from_pretrained(modelname)
        super().__init__(model, feature_extractor, modelname)

"""
# https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads
class BeitImageClassification(ImageProcessor):

    def __init__(self, modelname='facebook/deit-base-distilled-patch16-224'):
        feature_extractor = BeitFeatureExtractor.from_pretrained(modelname)
        model = BeitForImageClassification.from_pretrained(modelname)
        super().__init__(model, feature_extractor)

    def postprocess(self, output):
        return output
"""