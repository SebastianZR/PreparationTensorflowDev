import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.99):
            print("\n Felicidades Alcanzaste el 99 de precision")
            self.model.stop_training = True



def train_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizar datos
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Creamos la red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])
    model.compile(
        loss =  'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ["accuracy"]
    )

    callback = MyCallback()

    history = model.fit(
        x_train,
        y_train,
        validation_data = (x_test, y_test),
        epochs = 10,
        callbacks = [callback]
    )
    return history.epoch, history.history['accuracy'][-1]




if __name__ == '__main__':
    train_mnist()