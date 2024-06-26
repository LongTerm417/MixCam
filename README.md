# MixCam

This repository contains code to reproduce results from the paper:

[MixCam-Attack: Boosting the Transferability of Adversarial Examples with Targeted Data Augmentation](https://www.sciencedirect.com/science/article/abs/pii/S0020025523015037)

Sensen Guo, Xiaoyu Li, Peican Zhu, Baocang Wang, Zhiying Mu and Jinxiong Zhao

## Requirements
+ Python >= 3.6.5
+ 1.12.0 <= Tensorflow <= 1.15.0
+ Numpy >= 1.15.4
+ opencv >= 3.4.2
+ scipy > 1.1.0
+ pandas >= 1.0.1
+ imageio >= 2.6.1

## Qucik Start

### Prepare the data and models

You should download the [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and [pretrained models](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw) and place the data and pretrained models in dev_data/ and models/, respectively.

### MixCam

All the provided codes generate adversarial examples on inception_v3 model. If you want to attack other models, replace the model in `graph` and `batch_grad` function and load such models in `main` function.

#### Runing attack

Taking MixCam attack for example, you can run this attack as following:

```
python mi_mixcam.py 
```

#### Evaluating the attack

The generated adversarial examples would be stored in directory `./outputs`. Then run the file `simple_eval.py` to evaluate the success rate of each model used in the paper:

```
python simple_eval.py
```

## Acknowledgments

Code refers to [Admix](https://github.com/JHL-HUST/Admix) and [Grad-Cam](https://github.com/JHL-HUST/VT).

## Contact

Questions and suggestions can be sent to guosensen@mail.nwpu.edu.cn.
