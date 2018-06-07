import tensorflow as tf

import os
import dateutil.tz
import datetime
from  utils import mkdir_p
from utils import MnistData, celebA
from Gan import Gan
from Gan_celebA import Gan_celebA
import numpy as np

flags = tf.app.flags

flags.DEFINE_integer("OPER_FLAG" , 1 , "the flag of  opertion")
flags.DEFINE_integer("extend" , 1 , "contional value y")
flags.DEFINE_integer("celebA", 1, 'Choose celebA dataset')
flags.DEFINE_integer("VAE", 1, 'Choose VAE-GAN model')

FLAGS = flags.FLAGS


if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    celebA_FLAG = FLAGS.celebA

    if celebA_FLAG == 1:
        root_log_dir = "./tmp/logs/celebA_test"
        root_checkpoint_dir = "/tmp/gan_model/gan_model_celebA.ckpt"
        encode_z_checkpoint_dir = "/tmp/encode_z_model/encode_model_celebA.ckpt"
        encode_y_checkpoint_dir = "/tmp/encode_y_model/encode_model_celebA.ckpt"
        sample_path = "sample/celebA_gan"
        exp_name = "celebA_%s" % timestamp
    else:
        root_log_dir = "./tmp/logs/mnist_test"
        root_checkpoint_dir = "/tmp/gan_model/gan_model.ckpt"
        encode_z_checkpoint_dir = "/tmp/encode_z_model/encode_model.ckpt"
        encode_y_checkpoint_dir = "/tmp/encode_y_model/encode_model.ckpt"
        encode_vae_z_checkpoint_dir = "/tmp/encode_y_model/encode_model.ckpt"
        sample_path = "sample/mnist_gan"
        exp_name = "mnist_%s" % timestamp

    OPER_FLAG = FLAGS.OPER_FLAG

    # training GAN
    if OPER_FLAG == 0:

        build_model_flag = 0

    # training encoder z
    elif OPER_FLAG == 1:

        build_model_flag = 1

    # training encoder y
    elif OPER_FLAG == 2:

        build_model_flag = 2

    # using predict model
    elif OPER_FLAG == 3:

        build_model_flag = 3

    elif OPER_FLAG == 4:
        build_model_flag = 4

    elif OPER_FLAG == 5:
        build_model_flag = 5
    

    # batch_size = 1
    max_epoch = 20

    #for mnist train 62 + 2 + 10
    # sample_size = 512

    ## hyperparams
    dis_learn_rate = 0.0002
    gen_learn_rate = 0.0002

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    print('Creating directory...')
    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(sample_path)
    mkdir_p(encode_z_checkpoint_dir)
    mkdir_p(encode_y_checkpoint_dir)

    # MNIST dataset
    if celebA_FLAG == 0:

        batch_size = 64

        # sample_size = the length of z latent space
        sample_size = 64

        print ('Training MNIST dataset:')
        data , label = MnistData().load_mnist()

        IcGAN = Gan(batch_size=batch_size, max_epoch=max_epoch, build_model_flag = build_model_flag,
                      model_path=root_checkpoint_dir, encode_z_model=encode_z_checkpoint_dir,encode_y_model=encode_y_checkpoint_dir,
                      encode_vae_z_model=encode_vae_z_checkpoint_dir, data=data,label=label, extend_value=FLAGS.extend,
                      network_type="mnist", sample_size=sample_size,
                      sample_path=sample_path , log_dir = log_dir , gen_learning_rate=gen_learn_rate, dis_learning_rate=dis_learn_rate , info_reg_coeff=1.0)

    # celebA dataset
    else: 

        batch_size = 64
        sample_size = 100

        print ('Training celebA dataset:')
        data, label = celebA().load_celebA()

        print (data.shape)
        IcGAN = Gan_celebA(batch_size=batch_size, max_epoch=max_epoch, build_model_flag = build_model_flag,
                      model_path=root_checkpoint_dir, encode_z_model=encode_z_checkpoint_dir,encode_y_model=encode_y_checkpoint_dir,
                      data=data,label=label, extend_value=FLAGS.extend,
                      network_type="mnist", sample_size=sample_size,
                      sample_path=sample_path , log_dir = log_dir , gen_learning_rate=gen_learn_rate, dis_learning_rate=dis_learn_rate , info_reg_coeff=1.0)

    #start the training

    if OPER_FLAG == 0:

        print ("Training Gan")
        IcGAN.train()

    elif OPER_FLAG == 1:

        print ("Training encode for z")
        IcGAN.train_ez()

    elif OPER_FLAG == 2:

        print ("Training encode for Y")
        IcGAN.train_ey()

    elif OPER_FLAG == 3:
        print ("This is test of IcGAN")
        IcGAN.test_IcGAN()
    
    elif OPER_FLAG == 4:
        print("Training VAE-encoder")
        IcGAN.train_vae_ez()

    elif OPER_FLAG == 5:
        print("This is test of VAE-IcGAN")     

    else: 
        print('Wrong OPER_FLAG input')

