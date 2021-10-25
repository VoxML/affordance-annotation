import configparser
import json
import os.path

from tqdm import tqdm
from PIL import Image

from processor.imageprocessor import *


def process_files(config):
    annofile = config["HICODET"]["anno_list_json"]
    batch_size = int(config["PROCESSING"]["batch_size"])

    # Define Image Encoder Models. TODO: Move to Config
    image_processors = [DetrImageSegmentation()] #, DetrObjectDetection(), VitImageClassification(), DeitImageClassification()]

    # Load HICODET Json
    with open(annofile) as json_file:
        json_data = json.load(json_file)
    annosize = len(json_data)

    # Create Outputfolder if missing
    if not os.path.exists(config["PROCESSING"]["outputfolder"]):
        os.makedirs(config["PROCESSING"]["outputfolder"])

    # Process
    for image_processor in image_processors:
        print("Generate {tool} features".format(tool=image_processor.name))
        encoded_json = []

        # Batch Processing
        for anno_idx in tqdm(range(0, annosize, batch_size)):
            batch_anno = json_data[anno_idx:min(anno_idx + batch_size, annosize)]
            images = []
            for anno in batch_anno:
                images.append(Image.open(os.path.join(config["HICODET"]["hico_images"], anno["image_path_postfix"])))

            output = image_processor(images)

            for out_idx, out in enumerate(output.detach().tolist()):
                encoded_json.append({"global_id": batch_anno[out_idx]["global_id"], "encoder": out})

            #if anno_idx > 10:
            #    break

        toolname = image_processor.name.split("/")[-1]
        with open(os.path.join(config["PROCESSING"]["outputfolder"], toolname + ".json"), 'w') as outfile:
            json.dump(encoded_json, outfile, indent=4)
        #exit()


if __name__ == '__main__':
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    process_files(configp)
