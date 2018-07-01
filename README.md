# cs230-IcGAN-image-editing-tensorflow
Course project for CS230, implemented IcGAN on MNIST and celebA

1. Data:
Put MNIST and preprocessed celebA .npy files into data folder

2. Experiments:
Modify the corresponding params.json file inside the experiments/base_model/ folder.

To train on MNIST dataset:
`python main.py --OPER_FLAG <FLAG>`

To train on celebA dataset:
`python main.py --OPER_FLAG <FLAG> --celebA 1`

Replace `<FLAG>` with
`0` to train on IcGAN
`1` to train encoder z of IcGAN
`2` to train encoder y of IcGAN
`3` to test IcGAN

3. Results:
View the results in sample/celebA_gan or sampele/mnist_gan folder.

4. Tensorboard
run tensorboardx:

`tensorboard --logdir=<'path_to_log'>`

## Reference code

[DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
[IcGAN](https://github.com/zhangqianhui/ICGan-tensorflow)  



