# HICO-DAT Tool
 
## Setup

### Python enviroment
* https://docs.conda.io/en/latest/miniconda.html
* conda create -n "hicodet" python=3.8
* conda activate hicodet
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