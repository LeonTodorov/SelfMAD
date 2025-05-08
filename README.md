# Self-supervised Morphing Attack Detection method
![SelfMAD](https://github.com/user-attachments/assets/926e2319-880b-4eee-bbd3-99895966bcbd)

> [**SelfMAD: Enhancing Generalization and Robustness in Morphing Attack Detection via Self-Supervised Learning**](https://arxiv.org/abs/2504.05504),  
> Marija Ivanovska, Leon Todorov, Naser Damer, Deepak Kumar Jain, Peter Peer, Vitomir Štruc  
> *FG 2025 Preprint*

Examples of simulated general morphing attack artifacts observed during training:
![examples_img](https://github.com/user-attachments/assets/1afdd7b3-3be7-4bb9-bab9-e514bb00e32e)

# Author's Development Enviroment
* GPU: NVIDIA GeForce RTX 4090
* CUDA 12.5
  
# Dependencies
Python 3.10
- torch==2.3.1
- torchvision==0.18.1
- opencv-python==4.10.0.84
- albumentations==1.4.10
- albucore==0.0.16
- tqdm==4.66.4
- efficientnet-pytorch==0.7.1
- timm==1.0.8
  
Environment setup (conda):
```bash
conda env create -f ./conda/env.yml
conda activate selfMAD_env
```

# Datasets
## FaceForensics++ Dataset Structure
```
<path to FF++ dataset>    
└── <phase>      
    └── <method>     
        └── <video sequence ID>     
            └── *.png
```
- **\<phase\>** is one of:
  - `train`
  - `test`
  - `val`
  
- **\<method\>** is one of:
  - `Deepfakes`
  - `Face2Face`
  - `FaceShifter`
  - `FaceSwap`
  - `InsightFace`
  - `NeuralTextures`
  - `real`

- **\<video sequence ID\>** is in the range `[000, 999]`. 
  - The test/train/val split used is the same as the dataset author's.

**Dataset Source:** [FaceForensics GitHub Repository](https://github.com/ondyari/FaceForensics).

## SMDD Dataset Structure
```
<path to SMDD dataset>    
└── m15k_t    
    └── *.png
└── o25k_bf_t    
    └── *.png
```
**Dataset Source:**  [SMDD GitHub Repository](https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset).

## FRLL Dataset Structure
```
<path to FRLL dataset>    
└── <method>      
    └── *.jpg
```
- **\<method\>** is one of:
  - `morph_amsl`
  - `morph_facemorpher`
  - `morph_opencv`
  - `morph_stylegan`
  - `morph_webmorph`
  - `raw`

**Dataset Source:**  [Idiap research institute website](https://www.idiap.ch/en/scientific-research/data/frll-morphs)

## FRGC Dataset Structure
```
<path to FRGC dataset>    
└── <method>      
    └── *.jpg
```
- **\<method\>** is one of:
  - `morph_facemorpher`
  - `morph_opencv`
  - `morph_stylegan`
  - `raw` 

**Dataset Source:**  [Idiap research institute website](https://www.idiap.ch/en/scientific-research/data/frgc-morphs)

## FERET Dataset Structure
```
<path to FERET dataset>    
└── <method>      
    └── *.jpg
```
- **\<method\>** is one of:
  - `morph_facemorpher`
  - `morph_opencv`
  - `morph_stylegan`
  - `raw`

**Dataset Source:**  [Idiap research institute website](https://www.idiap.ch/en/scientific-research/data/feret-morphs)

## Landmarks
Generated landmarks have the same structure as the input datasets.
```bash
python preprocessing/dlib_lm.py \
-i <path_to_dataset_root> \
-o <path_to_output_root>
```
## Labels
Generated labels have the same structure as the input datasets.
```bash
CUDA_VISIBLE_DEVICES=* python preprocessing/face_labels.py.py \
-i <path_to_dataset_root> \
-o <path_to_output_root>
```
# Pre-trained model
We offer the pretrained HRNet-W18 model weights, which were used to achieve the results presented in the paper.  
You can download the weights from the following links:  
[HRNet-W18 Checkpoint](https://drive.google.com/file/d/1NOPppjuVxXLc4qu3Bs2AZQUErYzSdSG4/view?usp=sharing)  
[EfficientNet-b4 Checkpoint](https://drive.google.com/file/d/1fB7_u-VCwa8Mqo1i2c_DoL0E6vLDVcUG/view?usp=sharing)   
[EfficientNet-b7 Checkpoint](https://drive.google.com/file/d/1drjnA7SsM3Pk9QqYpqM7asbEQ4EhzE4A/view?usp=sharing)  
[Swin_base Checkpoint](https://drive.google.com/file/d/1kih2GrRrbQdhESarE5SmAlttK1J3Jnn-/view?usp=sharing)  
[ResNet-152 Checkpoint](https://drive.google.com/file/d/1GILjjbHBXsKg1ctxVdAnuBPTIAZz70l4/view?usp=sharing)  
 
# Inference
```bash
CUDA_VISIBLE_DEVICES=* python infer__.py \
-m <model> \
-p <path_to_checkpoint> \
-in <path_to_input_img>
```
We can use the provided pretrained model with some examples:
```bash
CUDA_VISIBLE_DEVICES=* python infer__.py \
-m hrnet_w18 \
-p ./checkpoints/hrnet_w18_checkpoint.tar \
-in ./images/morph.jpg
```
The output in terminal indicates the confidence that the image is a morph.

# Training
Before starting the training process, ensure that the dataset paths are properly configured in the `data_config.json` file.   
Additional training parameters can be tuned in the `train_config.json` file as needed.  
To start the training from the root directory of the project, run the following command:
```bash
CUDA_VISIBLE_DEVICES=* python train__.py \
-n <session_name>
```
Command-line arguments (specified in `train.py`) will override any default values set in the configuration files.
# Evaluation
Before starting the evaluation process, ensure that the dataset paths are properly configured in the `data_config.json` file.   
To start model evaluation from the root directory of the project, run the following command:
```bash
CUDA_VISIBLE_DEVICES=* python eval__.py \
-m <model> \
-p <path_to_checkpoint>
```
Using the provided pretrained model we can reproduce the results presented in the paper:
```bash
CUDA_VISIBLE_DEVICES=* python eval__.py \
-m hrnet_w18 \
-p ./checkpoints/hrnet_w18_checkpoint.tar
```

| Metric    | Value  |
|-----------|--------|
| **FRGC_fm (EER)** | 5.59   |
| **FRGC_cv (EER)** | 2.59   |
| **FRGC_sg (EER)** | 15.84  |
| **FERET_fm (EER)** | 3.19   |
| **FERET_cv (EER)** | 1.13   |
| **FERET_sg (EER)** | 18.14  |
| **FRLL_amsl (EER)** | 0.99   |
| **FRLL_fm (EER)** | 0.00   |
| **FRLL_cv (EER)** | 0.00   |
| **FRLL_sg (EER)** | 10.34  |
| **FRLL_wm (EER)** | 3.45   |
| **morphPIPE (EER)** | 5.89 |
| **GreedyDiM (EER)** | 7.60 |
| **MorCode (EER)** | 4.08 |
|-----------|--------|
| **Mean (EER)** | 5.63   |


# Citation
If our work contributes to your research, we would appreciate it if you cite our paper:
```bibtex
@misc{ivanovska2025selfmadenhancinggeneralizationrobustness,
      title={SelfMAD: Enhancing Generalization and Robustness in Morphing Attack Detection via Self-Supervised Learning}, 
      author={Marija Ivanovska and Leon Todorov and Naser Damer and Deepak Kumar Jain and Peter Peer and Vitomir Štruc},
      year={2025},
      eprint={2504.05504},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.05504}, 
}
```
