# PHNet
This is the official code for our paper:

> Boosting Convolution with Efficient MLP-Permutation for Volumetric Medical Image Segmentation

## Usage
The code is largely built on the [COVID-19-20 challenge baseline](https://github.com/Project-MONAI/tutorials/tree/main/3d_segmentation/challenge_baseline).

### Requirement
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117  
pip install monai==1.1.0  
pip install 'monai[nibabel,ignite,tqdm]'
```
For more details, please check [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/) and [MONAI installation guide](https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies).

### Data preparation
* [COVID-19-20](https://covid-segmentation.grand-challenge.org/Data/)
* [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* [MSD BraTS](http://medicaldecathlon.com/)
* [LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)

### Training & Evaluation
COVID-19-20: We provide the shell script below for training and testing. Then users can follow the instructions in [COVID-19-20 challenge baseline](https://github.com/Project-MONAI/tutorials/tree/main/3d_segmentation/challenge_baseline) to submit the predictions to challenge leaderboard. 
```
sh run_net.sh
``` 


Synapse: We follow the data split of [TransUNet](https://github.com/Beckschen/TransUNet?tab=readme-ov-file) and use [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet) for training and testing.  

MSD BraTS: We follow [VT-UNet](https://github.com/himashi92/VT-UNet) for training and testing.  

LiTS17: We follow [MedISeg framework](https://github.com/hust-linyi/MedISeg) for training and testing.

<font size=2>Note: We provide the network configuration of all datasets in config.py </font>  

### Acknowledgment
We thank [MONAI](https://github.com/Project-MONAI/tutorials), [nnUNet](https://github.com/MIC-DKFZ/nnUNet), [VT-UNet](https://github.com/himashi92/VT-UNet) and [MedISeg](https://github.com/hust-linyi/MedISeg) for the code we borrow from to conduct our experiments.

### Citation
Please cite the paper if you use the code.
```bibtex
TO BE ADDED
```