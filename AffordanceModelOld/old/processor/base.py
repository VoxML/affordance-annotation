class Processor:
    def __init__(self, model, feature_extractor, name):
        self.feature_extractor = feature_extractor
        self.model = model
        self.name = name


class ImageProcessor(Processor):
    def __init__(self, model, preprocessor, name):
        super().__init__(model, preprocessor, name)

    def preprocess(self, image):
        return self.feature_extractor(images=image, return_tensors="pt")

    def process(self, inputs):
        return self.model(**inputs)

    def postprocess(self, output):
        return output["encoder_last_hidden_state"]

    def __call__(self, image):
        inputs = self.preprocess(image)
        outputs = self.process(inputs)
        processed = self.postprocess(outputs)
        return processed


class TextProcessor(Processor):
    def __init__(self, model):
        super().__init__(model, None, None)