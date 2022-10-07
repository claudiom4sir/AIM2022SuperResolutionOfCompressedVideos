# AIM 2022 challenge on super-resolution of compressed videos

Pytorch implementation of the model proposed by the IVL team for the [AIM 2022 challenge on super-resolution of compressed videos](https://codalab.lisn.upsaclay.fr/competitions/5077) (ECCV22 Workshops). 


## Architecture
![_](https://github.com/claudiom4sir/SuperResolutionOfCompressedVideos/blob/main/NetworkDiagram.jpg)

## Requirements
Ubuntu 22.04, Python 3.7.13, CUDA 11.6.

For Python requirements, see requirements.txt.

## Dataset
The dataset for the challenge can be downloaded [here](https://github.com/RenYang-home/LDV_dataset).

## Train
The code for training will be released soon.

## Test
To reproduce the results, execute 
```python test_aim_challenge.py --data_path your_dataset_path```. By default, the script will output only frames 10, 20, 30 etc. If you want to output all frames, add ```--save_all True```.
The default output directory is "./Results/"

## Citations
```
@inproceedings{yang2022aim,
  title={AIM 2022 Challenge on Super-resolution of Compressed Image and Video: Dataset, Methods and Results},
  author={Yang, Ren and Timofte, Radu and others},
  booktitle={European Conference on Computer Vision Workshops},
  year={2022}
}
```
## Acknowledgements
Our method is based on [STDF](https://github.com/ryanxingql/stdf-pytorch).


## Contacts
For any question, please write an email to c.rota30@campus.unimib.it
