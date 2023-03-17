import zipfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def extract_images():
    local_zip = 'cats_and_dogs_filtered.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall()
    zip_ref.close()

def plt_loss_acc(history):
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    epochs = range(len(acc))

    plt.plot(epochs, acc, label='Traning Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title("Traning and validation Accuracy")

    plt.figure()

    plt.plot(epochs, loss, label='Traning loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title("Traning and validation Loss")

    plt.show()


def train_model():
    # Instanciar ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range = 40,
        width_shifet_range = 0.2,
        height_shifet_range=0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = False,


    )
    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0
    )
    train_dir = 'cats_and_dogs_filtered/train'
    test_dir = 'cats_and_dogs_filtered/validation'

    # Costruir generadores
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150,150)
    )

    # Contrutir red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    # Compilamos el modeloo
    model.compile(
        optimizer = RMSprop(learning_rate = 0.001),
        loss = "binary_crossentropy",
        metrics = ['accuracy']
    )

    # Entrenamos

    history = model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = 20,
        steps_per_epoch = 100 ,#2000imagenes / 20bathces
        validation_steps = 50 ,#1000imagenes / 20batches
    )

    plt_loss_acc(history)

    return model


if __name__ == '__main__':
    model = train_model()
    model.save('category3.h5')