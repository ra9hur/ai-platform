#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:48:01 2019

@author: raghu
"""

from keras.models import model_from_json
#from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#import os

# The following import and function call are the only additions to code required
# to automatically log metrics and parameters to MLflow.
import mlflow.keras

latent_dim = 100

'''
class_val = int(sys.argv[1])
goodrun_ref = int(sys.argv[2])
'''

class_val = 17
goodrun_ref = 9900


# ----------------- Below functions to generate fake images
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):

  # generate points in the latent space
  x_input = np.random.randn(latent_dim * n_samples)

  # reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):

  # generate points in latent space
  x_input = generate_latent_points(latent_dim, n_samples)

  # predict outputs
  X = g_model.predict(x_input)

  # create 'fake' class labels (0)
  y = np.zeros((n_samples, 1))
  return (X, y)
    



if __name__ == "__main__":
    

    mlflow.keras.autolog()
    
    mlflow.log_param('class_val', class_val)
    mlflow.log_param('goodrun_ref',goodrun_ref)


    
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





    # ----------------- Restore models from the local system directory
    # load json and create model
    g_file = base_dir + 'weights/gmodel.json'
    json_file = open(g_file, 'r')
    loaded_gmodel_json = json_file.read()
    json_file.close()
    g_model = model_from_json(loaded_gmodel_json)

    # load json and create model
    d_file = base_dir + 'weights/dmodel.json'
    json_file = open(d_file, 'r')
    loaded_dmodel_json = json_file.read()
    json_file.close()
    d_model = model_from_json(loaded_dmodel_json)

    # load json and create model
    gan_file = base_dir + 'weights/ganmodel.json'
    json_file = open(gan_file, 'r')
    loaded_ganmodel_json = json_file.read()
    json_file.close()
    gan_model = model_from_json(loaded_ganmodel_json)




    # ----------------- Restore saved weights
    g_file = base_dir + 'weights/generator_0' + str(class_val) + '_' + str(goodrun_ref) + '.h5'
    g_model.load_weights(g_file)

    d_file = base_dir + 'weights/discriminator_0' + str(class_val) + '_' + str(goodrun_ref) + '.h5'
    d_model.load_weights(d_file)

    gan_file = base_dir + 'weights/gan_0' + str(class_val) + '_' + str(goodrun_ref) + '.h5'
    gan_model.load_weights(gan_file)

    
    

    # ----------------- Evaluate fake images
    # Generate fake samples from the Generator
    # 20 samples are generated to evaluate Generator performance
    X_fake, _ = generate_fake_samples(g_model, latent_dim, n_samples=20)
    
    ("Number of fake samples generated: ", len(X_fake))
    
    
    '''
    for i in range(len(X_fake)):
        
        img = image.array_to_img(X_fake[i] * 255., scale=False)
        
        img.save(os.path.join(base_dir + 'images/evaluate/', str(class_val) + '_' + str(i) + '.png'))
    '''
    
    

    # Visualize fake images
    f, ax = plt.subplots(10, 2, figsize=(10,40))
    
    for i in range(10):
        for j in range(2):
            ax[i, j].imshow(X_fake[j+2*i])
    plt.tight_layout()
    eval_file = base_dir + 'images/evaluate/' + str(class_val) + '_' + str(goodrun_ref) + '.png'
    plt.savefig(eval_file)
    
    mlflow.log_param('Evaluate Generated Images', plt.show())
    
    
