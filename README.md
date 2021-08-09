# nCoVSegNet
COVID-19 Lung Infection Segmentation with A Novel Two-Stage Cross-Domain Transfer Learning Framework

<p align="center">
    <img src="images/overview.png" width="80%"/> <br />
</p>

## Requirements

- Python3
- Pytorch version >= 1.2.0.
- Some basic python packages such as Numpy, Pandas, SimpleITK.

## Data Preparation

- Please put the 2D CT images in the following directory: `./dataset/`, and organize the data as following structure:
  ``` 
     ├── dataset
        ├── train
           ├── image
               ├── 1.jpg, 2.jpg, xxxx
           ├── mask
               ├── 1.png, 2.png, xxxx
        ├── test
           ├── image
               ├── case01
                   ├── 1.jpg, 2.jpg, xxxx
               ├── xxxx
           ├── mask
               ├── case01
                   ├── 1.png, 2.png, xxxx
               ├── xxxx
   ```

## Training & Testing

- Train the nCoVSegNet:

  `python train.py`

- Test the nCoVSegNet:

  `python test.py`

  The results will be saved to `./Results`.

- Evaluate the result maps:

  You can evaluate the result maps using the tool in `./utils/evaluation.py`.

## Citation

Please cite the following paper if you use this repository in your reseach.

```
@inproceedings{liu2020covid19,
title={COVID-19 Lung Infection Segmentation with A Novel Two-Stage Cross-Domain Transfer Learning Framework},
author={Jiannan Liu, Bo Dong, Shuai Wang, Hui Cui, Dengping Fan, Jiquan Ma, Geng Chen},
booktitle={Medical Image Analysis},
year={2021}
}
```

## Acknowledgement

We implement this project based on the code of [Inf-Net](https://github.com/dengpingfan/inf-net) proposed by Fan et al.

# License

Our code is released under MIT License (see LICENSE file for details).

