#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:45:41 2019

@author: raghu
"""

import keras
from keras import layers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

import pickle

import os
import glob
import sys


# The following import and function call are the only additions to code required
# to automatically log metrics and parameters to MLflow.
import mlflow.keras


latent_dim = 100
height = 32
width = 32
channels = 3


iterations = int(sys.argv[1]) if len(sys.argv) >= 2 else 14100
class_val = int(sys.argv[2]) if len(sys.argv) >= 3 else 17
goodrun_ref = int(sys.argv[3]) if len(sys.argv) >= 4 else 12900


'''
iterations = 14000
class_val = 17
goodrun_ref = 12900
'''

# ----------------- Define GAN Generator Network
def define_generator(latent_dim):
    
    generator_input = keras.Input(shape=(latent_dim,))

    x = layers.Dense(256 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    
    x = layers.Reshape((16, 16, 256))(x)
    
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 5, padding = 'same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 5, padding = 'same')(x)
    x = layers.LeakyReLU()(x)
    
    # Produces a 32x32 1-channel feature map (shape of a Traffic-Sign image)
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    
    # Instantiates a generator model, which maps the input of shape (latent_dim) into an image of shape (32, 32, 3)
    generator = keras.models.Model(generator_input, x)
    print ("Generator Model Summary")
    generator.summary()
    mlflow.log_param('Generator Model Summary',generator.summary())
    
    return generator



# ----------------- Define GAN Discriminator Network
def define_discriminator(in_shape=(32,32,3)):

    discriminator_input = keras.Input(shape=(height, width, channels))
    x = layers.Conv2D(64, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 3, strides=2)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 3, strides=2)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(256, 3, strides=2)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = keras.models.Model(discriminator_input,x)
    
    print ("Discriminator Model Summary")
    discriminator.summary()
    mlflow.log_param('Discriminator Model Summary',discriminator.summary())
    
    #discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    discriminator_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    mlflow.log_param('Discriminator Optimizer',"Adam 0.0001")
    
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

    return discriminator



# ----------------- Define Adversarial Network
def define_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    
    gan = keras.models.Model(gan_input, gan_output)
    
    print ("Adversarial Network Model Summary")
    gan.summary()
    mlflow.log_param('GAN Model Summary',gan.summary())
    
    #gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    mlflow.log_param('GAN Optimizer',"Adam 0.0001")
    
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    return gan



# ----------------- Load pickled data
# load and prepare traffic-sign training images
def load_preprocess_real_data(base_dir, class_val):

    # Load pickled data
    
    training_file = base_dir + 'traffic-signs-data/train.p'
    testing_file = base_dir + 'traffic-signs-data/test.p'
    
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    x_train, y_train = train['features'], train['labels']
    x_test, y_test = test['features'], test['labels']
  
    x_train = x_train[y_train.flatten() == class_val]
    x_test = x_test[y_test.flatten() == class_val]
  
    x_train = np.concatenate([x_train, x_test])
  
    x_train = x_train.reshape((x_train.shape[0],)+(height, width, channels))
    
    x_train = x_train.astype('float32') / 255.
    
    return x_train



# ----------------- Train the network
# train the generator and discriminator
def train(generator, discriminator, gan, x_train, latent_dim, base_dir, class_val, step, iterations):

    image_gen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range = 0.01,
            zoom_range = [0.9, 1.25],
            horizontal_flip=True,
            data_format = 'channels_last')

    # If weight exists, iterations be greater than good weights from the previous run
    # If weights does not exist, iterations should start from zero to the number specified
    #step = 11900
    #iterations = 13000

    batch_size = 20
    d_loss = 0.0
    a_loss = 0.0
    
    n = len(x_train)
    r = n % batch_size
    x_train = x_train[:(n-r)]
        
    for X_batch in image_gen.flow(x_train, batch_size=batch_size, shuffle=True):

        if (len(X_batch) < batch_size):
          break

        #1. Draw random points in the latent space (random noise).
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        #2. Generate images with generator using this random noise.
        generated_images = generator.predict(random_latent_vectors)

        #3. Mix the generated images with real ones.
        combined_images = np.concatenate([generated_images, X_batch])

        #4. Train discriminator using these mixed images, with corresponding targets: either “real” (for the real images) or “fake” (for the generated images).
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        labels += 0.05 * np.random.random(labels.shape)

        # Randomise real images before training

        d_loss = discriminator.train_on_batch(combined_images, labels)

        #5. Draw new random points in the latent space.
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        #6. Train gan using these random vectors, with targets that all say “these are real images.”
        misleading_targets = np.zeros((batch_size, 1))

        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        if step % 100 == 0:

          # save weights of all three models periodically
          d_file = base_dir + 'weights/discriminator_%03d_%03d.h5' % (class_val, step)
          discriminator.save_weights(d_file)
          g_file = base_dir + 'weights/generator_%03d_%03d.h5' % (class_val, step)
          generator.save_weights(g_file)
          gan_file = base_dir + 'weights/gan_%03d_%03d.h5' % (class_val, step)
          gan.save_weights(gan_file)

          # Evaluate network performance
          print('Iter # ', step, '\t', 'discriminator loss:', d_loss, '\t', 'adversarial loss:', a_loss)

          img = image.array_to_img(generated_images[0] * 255., scale=False)
          img.save(os.path.join(base_dir + 'images/train/', 'generated_' + str(class_val) + '_' + str(step) + '.png'))

          img = image.array_to_img(X_batch[0] * 255., scale=False)
          img.save(os.path.join(base_dir + 'images/train/', 'real_' + str(class_val) + '_' + str(step) + '.png'))


        step += 1

        if step >= iterations:
          # we need to break the loop by hand because
          # the generator loops indefinitely
          break  

    mlflow.log_param('Discriminator Loss',d_loss)
    mlflow.log_param('Adversarial Loss',a_loss)




# ----------------- Helper Function
# Helper function to create a list object that holds images.
# These images are currently available in the local system folders
def load_train_images(image_dir, im_type):
    
    # Populate this empty image list
    im_list = []
    
    # Iterate through the folder
    path = os.path.join(image_dir, im_type + "*.png")
    print ("image path", path)
    
    for file in glob.glob(path):
            
        # Read in the image
        im = mpimg.imread(file)
            
        # Check if the image exists/if it's been correctly read-in
        if not im is None:
            im_list.append(im)

    return im_list



    


if __name__ == "__main__":
    

#    mlflow.keras.autolog()

    mlflow.log_param('iterations', iterations)
    mlflow.log_param('class_val', class_val)
    mlflow.log_param('goodrun_ref',goodrun_ref)
    
    # create the generator
    g_model = define_generator(latent_dim)
    
    # create the discriminator
    d_model = define_discriminator()
    
    # create the gan
    gan_model = define_gan(g_model, d_model)
    
    # If running on Google Drive, set gdrive to True
    gdrive = False

    if (gdrive):
        # Access files on the google drive
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=True)
        
        # Path variables to access project relevant files
        root_dir = "/content/gdrive/My Drive/"
        base_dir = root_dir + 'Colab Notebooks/SDC_Traffic_Sign/'
        
    else:
        base_dir = './'



    # ----------------- Save models to a local system
    # Serialize generator model to JSON
    gmodel_json = g_model.to_json()
    g_file = base_dir + 'weights/gmodel.json'
    with open(g_file, "w") as json_file:
        json_file.write(gmodel_json)

    # Serialize discriminator model to JSON
    dmodel_json = d_model.to_json()
    d_file = base_dir + 'weights/dmodel.json'
    with open(d_file, "w") as json_file:
        json_file.write(dmodel_json)

    # Serialize gan model to JSON
    ganmodel_json = gan_model.to_json()
    gan_file = base_dir + 'weights/ganmodel.json'
    with open(gan_file, "w") as json_file:
        json_file.write(ganmodel_json)




    # class_val should be between 1 to 43
    # If class_val is changed from the previous run, 
    # iterations should start from zero to get meaningful results
    # class_val = 17
    if class_val is None:
        print("Input Error: class_val required to train the network")
        sys.exit()
 
    if (1 <= class_val <= 43):
        # load image data
        dataset = load_preprocess_real_data(base_dir, class_val)
        print("Number of training samples: ", len(dataset))
    else:
        print("Input Error: class_val should be between 1 and 43")
        sys.exit()


    # ----------------- Restore weights from previous good runs
    # Restore saved weights
    #if (goodrun_ref is not None) and (goodrun_ref != 0):
    if (goodrun_ref is not None):

        g_file = base_dir + 'weights/goodrun/generator_0' + str(class_val) + '_' + str(goodrun_ref) + '.h5'
        print("g_file path: ", g_file)
        g_model.load_weights(g_file)
    
        d_file = base_dir + 'weights/goodrun/discriminator_0' + str(class_val) + '_' + str(goodrun_ref) + '.h5'
        d_model.load_weights(d_file)
    
        gan_file = base_dir + 'weights/goodrun/gan_0' + str(class_val) + '_' + str(goodrun_ref) + '.h5'
        gan_model.load_weights(gan_file)

    else:
        
        goodrun_ref = 0


    # train model
    if (iterations > goodrun_ref):

        print("Iterations: ", iterations)
        print("Class_val: ", class_val)
        print("Good run reference: ", goodrun_ref)
        input("Press Enter to continue, Cntrl C to exit ...")
        train(g_model, d_model, gan_model, dataset, latent_dim, base_dir, class_val, goodrun_ref, iterations)
        #print("within training loop")
    else:
        print("Input Error: Number of iterations should be greater than goodrun_ref")
        sys.exit()

    
    
    # ----------------- Visualize training images

    # Load generated images from the local system directory
    # These images were generated while training the network
    image_dir = base_dir + 'images/train/'
    real_all = load_train_images(image_dir, "real")
    generated_all = load_train_images(image_dir, "generated")
    print ("# of real images available for visualization: ", len(real_all))



    # Functionality to view real and generated images
    # These images were saved in the native system folders during network training
    n = len(real_all)
    
    real = real_all[10:]
    generated = generated_all[10:]
    
    # Plot the original image and the three channels
    f, ax = plt.subplots(10, 2, figsize=(10,40))
    
    ax[0,0].set_title('Real Samples', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax[0,1].set_title('Generated Samples', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    
    for i in range (10):
    
        ax[i,0].imshow(real_all[i])
    
        ax[i,1].imshow(generated_all[i])

    plt.tight_layout()
    train_file = base_dir + 'images/train/' + str(class_val) + '_' + str(iterations) + '.png'
    plt.savefig(train_file)
    
    mlflow.log_param('Evaluate Train Images', plt.show())



# Functionality to view real and generated images
# These images were saved in the native system folders during network training
def visualizeTrainingImages(base_dir):
    
    # Load generated images from the local system directory
    # These images were generated while training the network
    image_dir = base_dir + 'images/train/'
    real_all = load_train_images(image_dir, "real")
    generated_all = load_train_images(image_dir, "generated")
    print ("# of real images available for visualization: ", len(real_all))

    # Plot the original image and the three channels
    f, ax = plt.subplots(10, 2, figsize=(10,40))
    
    ax[0,0].set_title('Real Samples', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax[0,1].set_title('Generated Samples', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    
    for i in range (10):
    
        ax[i,0].imshow(real_all[i])
    
        ax[i,1].imshow(generated_all[i])

    plt.tight_layout()
    train_file = base_dir + 'images/train/' + str(class_val) + '_' + str(iterations) + '.png'
    plt.savefig(train_file)
    

        
        
        
