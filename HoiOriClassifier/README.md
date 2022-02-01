# HICO-DAT Tool
 
## Setup

### Python enviroment
* https://docs.conda.io/en/latest/miniconda.html
* conda create -n "habitat" python=3.8
* conda activate habitat
* install pytorch: https://pytorch.org/
* pip install -r requirements.txt
* (Maybe 1-2 libraries are still missing. In this case please add them or let me know.)

### Predict Results
* To use this tool, download the following models:
  * UPT Model from https://hessenbox-a10.rz.uni-frankfurt.de/dl/fi2jNX5TxJB1f5qbiZzRQnVU/robust-sweep-8_ckpt_41940_20.pt  (--hoi_model)
  * PoseContrast_ObjectNet3D_FewShot from https://github.com/YoungXIAO13/PoseContrast (--pose_model)

* python process_image_folder.py --input_folder "data/test_images" --output_file "results.json"
  * Predicts the results for every image in "test_images" and writes the results in "results.json"
  
#### Parameters
* input_folder: Path to Image Folder
* output_file: Output File
* device: Define GPU Device (-1 for CPU. Not tested.)
* box_score_thresh: Threshold for Object Detection
* hoi_model: Path to UPT Model
* pose_model: Path to ContrastPose Model