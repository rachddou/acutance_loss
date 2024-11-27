# Hybrid Training of Denoising Networks to Improve the Texture Acutance of Digital Cameras (SSVM'23)

[[Paper](https://link.springer.com/chapter/10.1007/978-3-031-31975-4_24)]

This repository is the official implementation of the paper *Hybrid Training of Denoising Networks to Improve the Texture Acutance of Digital Cameras*, by Raphaël Achddou, Yann Gousseau and Saïd Ladjal.

## Abstract
In order to evaluate the capacity of a camera to render textures properly, the standard practice, used by classical scoring protocols, is to compute the frequential response to a dead leaves image target, from which is built a **texture acutance metric**. In this work, we propose a mixed training procedure for image restoration neural networks, relying on both natural and synthetic images, that yields a strong improvement of this acutance metric without impairing fidelity terms. The feasibility of the approach is demonstrated both on the denoising of RGB images and the full development of RAW images, opening the path to a systematic improvement of the texture acutance of real imaging devices.

## Method

Inspired by the initial results shown in the seminal paper [*Synthetic images as a regularity prior for image restoration neural networks*](https://hal.science/hal-03186499/file/papier_SSVM%20%281%29.pdf)(SSVM'21), we extend the mixed training strategy to include an acutance loss, which diagram is the following : 
![](examples/schema_acutance.png)

## Prerequisities

### Requirements
To run this code, the following libraries are required:
- tensorboardx ==2.2
- torchvision==0.13.1
- pytorch==1.13.1
- scikit-image 
- numpy
- opencv 
- h5py

  Run the ```pip install -r requirements.txt``` to install all these libraries.


### Perceptual metric
Download the weights for the Pieapp metric with the .sh scripts in the ```PerceptualImageError/scripts``` directory.
### Natural Image datasets
- Download the test datasets and move them in ```datasets/test_sets``` :
    - [[Kodak24](https://link.springer.com/chapter/10.1007/978-3-031-31975-4_24)]
    - [[CBSD68](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original)]

- Download the Waterloo database , a natural image dataset :[[project page](https://ece.uwaterloo.ca/~k29ma/exploration/)] and put the images in ```datasets/src_imgs```

### Dead leaves image dataset

In order to create the synthetic image dataset made of dead leaves images, run the following command: 
```
python3 launcher_data_generation.py --path [PATH_NAME] --size 2000 --rmin 1 --rmax 2000 --number [NUMBER_IM] --nb_p [NUMBER_PROCESS] --ds --natural --color_path datasets/src_imgs
```

- ```[PATH_NAME]``` : specify the name you want for your synthetic set 
- ```[NUMBER_IM]``` : the number of generated image per subprocess
- ```[NUMBER_PROCESS]``` : the number of suprocesses you want to launch in parallel

The total number of images created is the product  : $number_{process}\times number_{im} $.

<span style="color:red">**WARNING:** </span>
The dict.npy file takes quite a lot of RAM. In order to reduce this, you can reduce the maximal reduce when calling ```disk_dict``` in the ```launcher_data_generation.py``` file. Also, make sure that ```[NUMBER_PROCESS]``` is lower than the number of CPU cores available.

### Dataset preprocessing

Run the following commands to pre-process all the images and store them in a ```.h5``` file:
```
python3 prepare_patches.py --stride 35 --trainset_dir datasets/src_imgs/ --m 500000  --fn train_natural/ val/ 
python3 prepare_patches.py --stride 50 --trainset_dir datasets/src_imgs/ --m 250000  --fn train_dead_leaves/ val/ 
```
This command should create ```.h5``` files in ```datasets/h5files/train_natural/``` and ```datasets/h5files/train_dead_leaves/``` that can be used for training.

## Training
To train the network with the acutance loss, run the following command : 

```
python3 train.py --bs 100 --e 150 --milestone 60 90 120 --log_dir train_acutance_lambda_20 --new_mtf  --l_ac 20 --grad_clip --single_dim_acutance  --save_every_epochs 10
```

## Testing

To test the trained models, run the following command:
- for natural image datasets : 
```
python3 launcher_test.py --input datasets/test_sets/Kodak24/ --p TRAINING_LOGS/train_acutance_lambda_20/net.pth  --noise_sigma 25 --pieapp

```
- for the synthetic dead leaves image datasets : 
```
python3 test_dataset.py --input datasets/test_sets/Kodak_24_dead_leaves/ --p TRAINING_LOGS/train_acutance_lambda_20/net.pth  --noise_sigma 25 --acutance --pieapp
```


## Citation 

If you make use of our work, please cite our paper:

```
@inproceedings{achddou2023hybrid,
  title={Hybrid Training of Denoising Networks to Improve the Texture Acutance of Digital Cameras},
  author={Achddou, Rapha{\"e}l and Gousseau, Yann and Ladjal, Sa{\"\i}d},
  booktitle={International Conference on Scale Space and Variational Methods in Computer Vision},
  pages={314--325},
  year={2023},
  organization={Springer}
}
```
