# UPT: Unary&ndash;Pairwise Transformers

## SetUp
* Install pocket (needed for UPT): https://github.com/fredzzhang/pocket
* Transformers with Pytorch: https://huggingface.co/docs/transformers/installation
* (WandB: https://docs.wandb.ai/quickstart ; Can also be omitted, but must then also be removed from the project)

## Main Differences to UPT
### Model
AffordanceUPT can be found in upt.py 
Changes worth mentioning:
* Supporting two classes: Gibsonian and Telic
* Modularization to support pretrained Huggingface DETR / other object detection models.

### Data Loader
DataLoader for annotated HicoDet Data can be found in hicodet/hicodet.py
Adapted to support Gibsonian / telic annotations from "textual_annotations.csv" (see HicoDet Dataset)
The Dataset gets initiated in: utils.py:79
This is important for later usage (see Usage).

## Usage
### Hyperparameter / Arguments
Hyperparameters were found via wandb.
The old default hyperparameters of upt were commented out.
The main hyperparameters:
* data-root: Path for HicoDet Data. See "HicoDet Dataset" for more information
* output-dir: where to save the trained model.

### HicoDet Dataset
The folder under "data-root" should contain: 
* hico_20160224-det (folder): the original HicoDet dataset
* instances_train2015.json: Original HicoDet Annotations
* instances_test2015.json: Original HicoDet Annotations
* textual_annotations.csv: To be found in this git (affordance-annotation/Annotation Data/textual_annotations.csv)

### Train
* python main.py (change "data-root" and "output-dir")

### Evaluation
* CUDA_VISIBLE_DEVICES=0 python main.py --world-size 1 --eval --resume "path_to_model.pt"

### Notes
* utils.py:87 let you change the evaluations for different objectsets. 
  * but it also needs a little more preparation time. -> Will be added
* wandb may need a little cleaning ...

