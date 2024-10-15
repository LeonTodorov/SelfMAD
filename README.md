# Self-supervised Morphing Attack Detection method
![SelfMAD](https://github.com/user-attachments/assets/d59c1b00-da81-4c57-b6b7-112dbc9292f6)
> [**SelfMAD: Enhancing Generalization and Robustness in Morphing Attack Detection via Self-Supervised Learning**](),  
> Marija Ivanovska, Leon Todorov, Naser Damer, Deepak Kumar Jain, Peter Peer, Vitomir Štruc  
> *FG 2025 Preprint*


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
conda env create -f env.yml
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
<path to FRLL dataset>    
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
Use the same structure as their respective datasets.
TODO
## Labels
Use the same structure as their respective datasets.
TODO
# Inference
TODO
# Training
Before starting the training process, ensure that the dataset paths are properly configured in the `data_config.json` file.   
Additional training parameters can be tuned in the `train_config.json` file as needed.  
To start the training from the root directory of the project, run the following command:
```bash
python src/train__.py -n <session_name>
```
Command-line arguments (specified in `train.py`) will override any default values set in these configuration files.
# Evaluation
Before starting the evaluation process, ensure that the dataset paths are properly configured in the `data_config.json` file.   
To start model evaluation from the root directory of the project, run the following command:
```bash
python src/eval__.py -m <model> -p <path_to_checkpoint>
```
# Citation
TODO
