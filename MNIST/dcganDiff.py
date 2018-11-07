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
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import Concatenate
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



# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
# parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])#
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
args = parser.parse_args()
K.set_image_dim_ordering('th')



# Read MNIST DATA x_train is (60000,28,28) x_test = (10000,28,28)
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#get all the ones data
ones_train  = x_train[np.where(y_train == 1)]
ones_train = (ones_train.astype(np.float32) - 127.5)/127.5
ones_train = x_train[:,np.newaxis,:, : ]

input_shape = (1,img_cols, img_rows)
# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
# model1 = Model1(input_tensor=input_tensor)
# model2 = Model2(input_tensor=input_tensor)
# model3 = Model3(input_tensor=input_tensor)

# # init coverage table

# model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)


# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim  = 100
adam = Adam(lr=args.step, beta_1=0.5)
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
#print(x.shape)
# in deepexplore code       input_shape = (img_rows, img_cols, 1) tf
# in gan code               input_shape = (1,img_rows, img_cols) th
discriminator = Sequential()

discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)
discriminator.trainable = False

generator.load_weights('./models/gan_generator_epoch_1.h5')
print(bcolors.OKBLUE + 'Model2 loaded' + bcolors.ENDC)

discriminator.load_weights('./models/gan_discriminator_epoch_1.h5')
print(bcolors.OKBLUE + 'Model2 loaded' + bcolors.ENDC)

ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)
#print(gan.summary())
#
# gen_img = generator.predict()
#
# orig_img = gen_img.copy()

def cusloss(y_true, y_pred):
    return  K.sum(y_pred)
        #K.binary_crossentropy(y_true[...,1], y_pred), axis=-1)
adam2 = Adam(lr = args.step, beta_1=0.5)
#actually we don't care all output of model1,2,3 but only the category that we focus on
#model = Model( x)
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
# neu1_output = K.mean(model1.get_layer(layer_name1).output[..., index1])
# neu2_output = K.mean(model2.get_layer(layer_name2).output[..., index2])
# neu3_output = K.mean(model3.get_layer(layer_name3).output[..., index3])
#

m1output = model1(x)# * args.weight_diff
m2output = model2(x)
m3output = model3(x)
#print(m1output.shape, m2output.shape, m3output.shape)


def multiply(x):
    return args.weight_diff*x
m1outputass = Lambda(multiply)(m1output)
outputs =  concatenate([m1outputass, m2output, m3output])

def myFunc(x):
    return x[:,:1]

la  = Lambda(myFunc,output_shape=(3,1))(outputs)
#print("m1output shape: "+str(m1output.shape))

#print(m1output.shape)

#neu_output = [neu1_output, neu2_output,neu3_output]
#lamlayer =Lambda(lambda x:x)
#outlayer = Concatenate([m1output,m2output,m3output])
DiffNetwork = Model(input = ganInput, outputs = la)
DiffNetwork.compile(loss =cusloss, optimizer = adam2)#, loss_weights=[args.weight_diff,1,1])

#print(DiffNetwork.summary())
#, loss_weights=[1, args.weight_nc])


# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = x_test.shape[0] / batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in xrange(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        diffCount = 0
        for _ in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch =ones_train[np.random.randint(0, ones_train.shape[0],
                                                            size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print(generatedImages.shape)
            # #to verify crotness
            # generatedImages = generatedImages.reshape((generatedImages.shape[0],28,28,1))
            # print(generatedImages.shape, imageBatch.shape)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2 * batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)


            # Train generator
            #noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)
            #print(generatedImages.shape)
            label1 = np.argmax(model1.predict(generatedImages), axis=1)
            #print(label1)
            label2 = np.argmax(model2.predict(generatedImages),axis= 1)
            label3 = np.argmax(model3.predict(generatedImages),axis= 1)

            labeltotrain = (np.abs(label1 - label2) + np.abs(label2 - label3))!=0
            diffCount = labeltotrain.sum()
            diffTrain = np.repeat(labeltotrain, randomDim).reshape((batchSize,randomDim))
            diffTrain = np.extract(diffTrain, noise).reshape((diffCount, randomDim))
            # yGen here doesn't matter
            generator.trainable = True
            #DiffNetwork.train_on_batch(diffTrain, [np.zeros(diffCount)])
            print("Diffrate for this batch:" +str(diffCount) +' '+ str(batchSize))
        # for adversarial image generation


        # final_loss = K.mean(layer_output)
        #
        # # we compute the gradient of the input picture wrt this loss
        # grads = normalize(K.gradients(final_loss, input_tensor)[0])
        #
        # # this function returns the loss and grads given the input picture
        # iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])


        # Store loss of most recent batch from this epoch
        # dLosses.append(dloss)
        # gLosses.append(gloss)
        #
        if e == 1 or e % 5 == 0:
            plotGeneratedImages(e)
            #saveModels(e)

    # Plot losses from every epoch
    # plotLoss(e)


if __name__ == '__main__':
    train(50, 128)

