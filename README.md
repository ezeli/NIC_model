# NIC model
 - A pytorch implementation of "[Show and Tell: A Neural Image Caption Generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html)".
 - Add SCST training from "[Self-critical Sequence Training for Image Captioning](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html)".
 - Refer to [ruotianluo](https://github.com/ruotianluo/ImageCaptioning.pytorch).

## Environment
 - Python 3.7
 - Pytorch 1.3.1

## Usage
### 1. Preprocessing
Extract image features and process coco captions data (from [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)) through `preprocess.py`. Need to adjust the parameters, where `resnet101_file` comes from [here](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM).
### 2. Training
 - First adjust the parameters in `opt.py`:
    - train_mode: 'xe' for pre-training, 'rl' for fine-tuning
    - learning_rate: '4e-4' for xe, '4e-5' for rl
    - resume: resume training from this checkpoint. required for rl.
    - other parameters can be modified as needed.
 - Run:  
    `python train.py`  
    checkpoint save in `checkpoint` dir, test result save in `result` dir.

## Result
### Evaluation metrics

### Examples
