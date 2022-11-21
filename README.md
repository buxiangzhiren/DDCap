# Exploring Discrete Diffusion Models for Image Captioning.


## Official implementation for the paper ["Exploring Discrete Diffusion Models for Image Captioning"]()




## Description  





## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
You can use [docker](https://hub.docker.com/r/zixinzhu/pytorch1.9.0). Also, you can create environment and install dependencies:
```
conda env create -f environment.yml
```
or
```
bash install_req.sh
```

## COCO training

Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing).

Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).

### Microsoft COCO
```
│MSCOCO_Caption/
├──annotations/
│  ├── captions_train2014.json
│  ├── captions_val2014.json
├──train2014/
│  ├── COCO_train2014_000000000009.jpg
│  ├── ......
├──val2014/ 
│  ├── COCO_val2014_000000000042.jpg
│  ├── ......
```

### Prepare evaluation
Change the work directory and set up the code of evaluation :
```
cd ./captioneval/coco_caption
bash ./get_stanford_models.sh
```
### Prepare evaluation
```
python train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/
```

## Citation
If you use this code for your research, please cite:
```
```




## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

## Contact
For any inquiry please contact us at our email addresses: ron.mokady@gmail.com or amirhertz@mail.tau.ac.il.


