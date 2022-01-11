import os
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests


folder = "D:/Corpora/HICO-DET/hico_20160224_det/images/test2015"
files = ["HICO_test2015_00000001.jpg", "HICO_test2015_00000002.jpg", "HICO_test2015_00000003.jpg"]

images = []
for file in files:
    path = os.path.join(folder, file)
    images.append(Image.open(path).convert("RGB"))

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')

inputs = feature_extractor(images=images, return_tensors="pt")
