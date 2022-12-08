# HICO-DET Tool
Used for visualization of Hico-Det data in combination with PoseContrast (https://github.com/YoungXIAO13/PoseContrast)
Not necessary for the final work, but leave it here for now uas work reasons.

## Setup

### Python enviroment
* https://docs.conda.io/en/latest/miniconda.html
* conda create -n "hicodet" python=3.8
* conda activate hicodet
* install pytorch: https://pytorch.org/
* pip install -r requirements.txt

### Convert Hico-Det to json
* Download HICO-DET.
* Copy "anno.mat" & "anno_bbox.mat" into "data/hico_mat" or change "anno_file" & "anno_bbox_file" in "config.ini".
* run HicoDet2JsonConverter.py.

### Visualize Images
* Change "hico_images" in config.ini to the hico-det image folder.
* Run HicoDetVisualizer.py.
* Change ImageID in Line 82 for other Images.
* TODO: Support Image Names.

### Pose Data Extracted with AlphaPose (https://github.com/MVIG-SJTU/AlphaPose)
* To visualize images with poses, download the following zip and copy the files under data/poses.
* https://hessenbox-a10.rz.uni-frankfurt.de/getlink/fi7NNa5DLocv3DcUy5HPd3Zz/poses.zip
* In the config.ini can you referenz on the different pose datasets.
* (Following models were used: Fast Pose (DUC) -	ResNet152 (MSCOCO), Fast Pose - ResNet50 (Halpe 26), Fast Pose - ResNet50 (Halpe 136))
