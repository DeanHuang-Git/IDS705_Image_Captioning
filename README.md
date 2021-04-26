# Comparison of Pre-Inject Architecture and Merge Architecture for Image Captioning
Final Project for <a href="https://kylebradbury.github.io/ids705/index.html"> IDS 705: Principles of Machine Learning <a/> <br />

## Contents
   - [References](#references)
   - [Abstract](#abstract)
   - [Architecture Framework](#architecture-framework)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Instructions for Running Merge Architecture](#instructions-for-running-merge-architecture)
   - [Instructions for Running Pre-inject Architecture](#instructions-for-running-preinject-architecture)
      + [Before training](#before-training)
      + [Preprocess](#preprocess)
      + [Training Process](#training-process)
      + [Evaluation metrics](#evaluation-metrics)
      + [Testing Process](#testing-process)

### References
*Reference for merge architecture and training code: <a href="machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/">How to Develop a Deep Learning Photo Caption Generator from Scratch</a>. <br />
Reference for pre-inject architecture and training code: <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning">a-PyTorch-Tutorial-to-Image-Captioning</a>. <br />
Reference Paper <a href="https://arxiv.org/pdf/1502.03044.pdf">Show, Attend and Tell</a> <br />
Data comes from: <a href="https://www.kaggle.com/adityajn105/flickr8k?select=Images">Flickr 8k Dataset</a> <br />*

### Abstract
Blind and visually impaired people (BVIP) have consistently expressed their frustration on the lack of accessibility to visual content via social media. The goal of this project is to build high-performance encoder-decoder image captioning models using Pre-inject Architecture and Merge Architecture to assist BVIP in comprehending the message and social context of the images. The two image encoders we decided to implement were VGG and ResNet, and the language model we decided to implement was LSTM. Our experiment results showed that Pre-Inject Architecture outperformed the Merge Architecture by a large margin. In addition, the total training duration for Pre-inject Architecture was approximately 13 times higher than the Merge Architecture.

### Architecture Framework

#### Merge Architecture

*<a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/blob/main/10_code/RESNET(152).ipynb'> ResNet (152)</a> <br />
<a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/blob/main/10_code/RESNET.ipynb'> ResNet (50)</a> <br />
<a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/blob/main/10_code/VGG.ipynb'> VGG16</a> <br />*

#### Pre-inject Architecture

*<a href=https://github.com/DeanHuang-Git/IDS705_Image_Captioning/tree/main/10_code/pre-inject> ResNet (101)</a> <br />*

### Hyperparameter Tuning 

**Merge Architecture's detailed results locate at <a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/tree/main/30_results'> 30_results directory</a>. <br />
Pre-inject Architecture's detailed results locate at https://github.com/wkhalil/Image-Caption/tree/main/test_results**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Instructions for Running Merge Architecture
1. clone the current repository to servers
```
$ git clone git@github.com:DeanHuang-Git/IDS705_Image_Captioning.git
```
2. before running the files below, make sure the data is downloaded and saved in the right path.

*<a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/blob/main/10_code/RESNET(152).ipynb'> ResNet (152)</a> <br />
<a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/blob/main/10_code/RESNET.ipynb'> ResNet (50)</a> <br />
<a href='https://github.com/DeanHuang-Git/IDS705_Image_Captioning/blob/main/10_code/VGG.ipynb'> VGG16</a> <br />*

### Instructions for Running Preinject Architecture
#### Before Training
1. clone the current repository to servers
```
$ git clone git@github.com:wkhalil/Image-Caption.git
```
2. **before training & testing: make sure all libraries are installed with compatible versions (requirements.txt for reference).**
  - using the interpreter with compatible environment to open the project.
  - if using MacOS Catalina, please run the following commands to avoid potential bug. https://stackoverflow.com/questions/48290403/process-finished-with-exit-code-134-interrupted-by-signal-6-sigabrt
3. check for directory tree (missing folders can be created manually, most with place_holder.txt) <br />
download <a href="https://www.kaggle.com/adityajn105/flickr8k?select=Images">Flickr 8k Dataset</a> and store the folder with images in project as './inputs/Images'. <br />
directory necessary before training:
```
.
├── 10_code
├── 20_intermediate_files
├── 30_results
├── 40_docs
├── README.md
└── requirements.txt
```
---
#### Preprocess
```
$ python create_input_files.py
```
(only once for generating intermediate data files)
---
#### Training Process
1. There are two choices for training, if training without previous checkpoints, **set checkpoints=None in config.py**. <br />
Another one is training based on current best model (**default, obtain best model: <a href="https://drive.google.com/drive/folders/1E3W1wKbhV20FyBfRfTXfRcjAVjoIQavp?usp=sharing">latest model checkpoints<a/>**).
2. run the train.py
```
$ python train.py
```
---
### Evaluation metrics
first install <a href="https://github.com/salaniz/pycocoevalcap"> pycocoevalcap <a/> for CIDER, SPICE metrics.(Problems still exist for SPICE after installing following <a href="https://github.com/jiasenlu/coco-caption">instructions<a/>, and others also met <a href="https://github.com/jiasenlu/NeuralBabyTalk/issues/9">the same problem<a/>)
```
$ python eval.py
```
---
### Testing Process
All generated captions would be stored under test_results directory. <br />
Three choices:
1. If randomly generated one caption from all inputs
```
$ python caption.py
```
2. If randomly generated multiple captions from all inputs <br />
ex. randomly select 6 images to generate captions
```
$ python caption.py --num = 6
```
3. generate caption for specified image & specified model & specified path to save<br />
ex.
```
$ python caption.py  --word_map='./inputs/Intermediate_files/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json' --beam_size=5 --img='./inputs/Images/2877424957_9beb1dc49a.jpg' --save='./test_results/gen_3' --model='./model_history/best_0417_20.pth.tar'
```
