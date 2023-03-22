import urllib.request
import keras.layers
import tensorflow as tf
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

# Crear un callback para detener el entrenamiento cuando llegue al 85%
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=97):
            print("\n Haz alcanzado el 97 % de precicion.")
            self.model.stop_traning = True

def get_data():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse_or_human.zip')
    local_zip = 'horse_or_human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
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
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=30,
        class_mode='binary',
        target_size=(300, 300)
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        batch_size=30,
        class_mode='binary',
        target_size=(300, 300)
    )
    # Costruimosla red Neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    # Entrenamos

    callback = MyCallback()

    history = model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = 40,
        steps_per_epoch = 527/30, # 1000 imagenes / 30 batches
        validation_steps= 128/30 , # 256 imagenes / 30 batches
        callbacks = [callback]
    )
    return history.epoch, history.history['accuracy'][-1]


if __name__ == '__main__':
    model = solution_model()
    #model.save("category3ejercicio.h5")


