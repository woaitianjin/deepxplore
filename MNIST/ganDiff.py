'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse
from scipy.misc import imsave


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten,Lambda
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras import initializers
from keras.optimizers import Adam

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
random.seed(100)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
# parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])#
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('digit', help="digits to retrieve", type=int)
parser.add_argument('ratio', help="params for diff/gan ratios in iterations", type=int)
args = parser.parse_args()
K.set_image_dim_ordering('th')



# Read MNIST DATA x_train is (60000,28,28) x_test = (10000,28,28)
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#get all the ones data
ones_train  = x_train[np.where(y_train == args.digit)]
ones_train = (ones_train.astype(np.float32) - 127.5)/127.5
ones_train = ones_train.reshape((ones_train.shape[0], 28*28))


# load multiple models sharing same input tensor
# model1 = Model1(input_tensor=input_tensor)
# model2 = Model2(input_tensor=input_tensor)
# model3 = Model3(input_tensor=input_tensor)

# # init coverage table

# model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)


# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim  = 100
adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(256, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)


#print(x.shape)
# in deepexplore code       input_shape = (img_rows, img_cols, 1) tf
# in gan code               input_shape = (1,img_rows, img_cols) th

generator.load_weights('./models/gan_generator_digit_%d_epoch_%d.h5'% (args.digit, 1))
print(bcolors.OKBLUE + 'Model params generator loaded' + bcolors.ENDC)

discriminator.load_weights('./models/gan_discriminator_digit%d_epoch_%d.h5'% (args.digit, 1))
print(bcolors.OKBLUE + 'Model params discriminator loaded' + bcolors.ENDC)

ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)
#print(gan.summary())
#
# gen_img = generator.predict()
#
# # orig_img = gen_img.copy()

def cusloss(y_true, y_pred):
    return  K.mean(-K.log(y_pred))
adam2 = Adam(lr = args.step, beta_1=0.5)
#actually we don't care all output of model1,2,3 but only the category that we focus on
#model = Model( x)
input_shape = (1, img_rows, img_cols)
# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)
model1 = Model1(input_tensor= input_tensor)

model2 = Model2(input_tensor= input_tensor)
model3 = Model3(input_tensor= input_tensor)

model1.trainable = False
model2.trainable = False
model3.trainable = False

orig_label = 1

# layer_name1, index1 = neuron_to_cover(model_layer_dict1)
# layer_name2, index2 = neuron_to_cover(model_layer_dict2)
# layer_name3, index3 = neuron_to_cover(model_layer_dict3)
#

# neu2_output = K.mean(model2.get_layer(layer_name2).output[..., index2])
# neu3_output = K.mean(model3.get_layer(layer_name3).output[..., index3])
#
xreshaped = Reshape((1,28,28))(x)


m1output = model1(xreshaped)# * args.weight_diff
neu1_output = model1.get_layer("block2_pool1").output

m2output = model2(xreshaped)
m3output = model3(xreshaped)
print(m1output.shape)

def sle(x):
    return tf.gather(x, 1, axis =1)

o1 = Lambda(sle,name = "oo1")(m1output)
o2 = Lambda(sle,name = "oo2")(m2output)
o3 = Lambda(sle,name = "oo3")(m3output)

o11 = Reshape((1,), name = "o1")(o1)
o22 = Reshape((1,), name = "o2")(o1)
o33 = Reshape((1,), name = "o3")(o1)
print(o1.shape)
DiffNetwork = Model(outputs = [o11, o22, o33], input = ganInput)
DiffNetwork.compile(loss = {"o1":"binary_crossentropy",
                            "o2":"binary_crossentropy",
                            "o3":"binary_crossentropy",
                            },
                    optimizer = adam2,
                    loss_weights={"o1":args.weight_diff, "o2":1,"o3":1})
print(DiffNetwork.summary())


# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)

    label1 = np.argmax(model1.predict(generatedImages.reshape((examples, 1, 28, 28))), axis=1)
    label2 = np.argmax(model2.predict(generatedImages.reshape((examples, 1, 28, 28))), axis=1)
    label3 = np.argmax(model3.predict(generatedImages.reshape((examples, 1, 28, 28))), axis=1)
    labeltotrain = (np.abs(label1 - label2) + np.abs(label2 - label3)) != 0
    diffCount = labeltotrain.sum()
    print(diffCount, "for epoch ",epoch )

    generatedImages = generatedImages.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        if label1[i] == label2[i] == label3[i]:
            continue
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_digit_%d_rate_%f_ratio_%f_weight%f_epoch_%d.png' % (args.digit,args.step,args.ratio, args.weight_diff, epoch))


# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = x_test.shape[0] / batchSize
    #batchCount = 25
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in xrange(1, epochs+1):
        diffEpoch = 0
        print('-'*15, 'Epoch %d' % e, '-'*15)
        #batchCount = 10
        for _ in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = ones_train[np.random.randint(0, ones_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2 * batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)
            #print(generatedImages.shape)

            label1 = np.argmax(model1.predict(generatedImages.reshape((batchSize,1,28,28))),axis=1)
            label2 = np.argmax(model2.predict(generatedImages.reshape((batchSize,1,28,28))),axis= 1)
            label3 = np.argmax(model3.predict(generatedImages.reshape((batchSize,1,28,28))),axis= 1)

            labeltotrain = (np.abs(label1 - label2) + np.abs(label2 - label3))!=0
            diffCount = labeltotrain.sum()
            diffEpoch+= diffCount
            diffTrain = np.repeat(labeltotrain, randomDim).reshape((batchSize,randomDim))
            diffTrain = np.extract(diffTrain, noise).reshape((diffCount, randomDim))
            #print(diffTrain.shape)
            if diffCount == 0 or _%args.ratio != 0:
                continue
            diffLoss = DiffNetwork.train_on_batch(diffTrain, {"o1":np.ones(diffCount),"o2":np.zeros(diffCount),"o3":np.zeros(diffCount)})
            print(diffLoss)

        print("Diffrate for this epoch: " +str(diffEpoch)+"/"+str(batchCount*batchSize))

        # for adversarial image generation

        # Store loss of most recent batch from this epoch
        # dLosses.append(dloss)
        # gLosses.append(gloss)
        if e < 5 or e % 5 == 0:
            plotGeneratedImages(e)
        #saveModels(e)

    # Plot losses from every epoch
    # plotLoss(e)


if __name__ == '__main__':
    #print ("he")
    train(30, 128)

