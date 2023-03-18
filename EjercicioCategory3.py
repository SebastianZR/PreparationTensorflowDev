import urllib.request

import tensorflow as tf
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

# Crear un callback para detener el entrenamiento cuando llegue al 85%
def myCallback():
    pass

def get_data():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse_or_human.zip')
    local_zip = 'horse_or_human.zip'
    zip_ref = zipfile.ZipFile.(local_zip , 'r')
    zip_ref.extractall('data/horse_or_human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('data/testdata/')
    zip_ref.close()

def solution_model():
    train_dir = 'data/horse_or_human'
    validation_dir = 'data/testdata'
    print(f' Total training horse images : {len(os.listdir(os.path.join(train_dir, "horses")))}')
    print(f' Total training horse images : {len(os.listdir(os.path.join(train_dir, "humans")))}')
    print(f" Total validation humans images : {len(os.listdir(os.path.join(validation_dir, 'horses')))}")
    print(f" Total validation humans images : {len(os.listdir(os.path.join(validation_dir, 'humans')))}")

    # Cargamos los datos mediante ImageDataGenerator


