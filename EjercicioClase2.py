import tensorflow as tf

class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.97):
            print("\n Felicidades Alcanzaste el 97 de precision")
            self.model.stop_training = True

def train_fmnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Normalizamos datos
    # Dividimos entre 255 para que 1 sea blanco y 0 sea negro
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Creamos red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),# Agregamos la primer capa como Flatten para que divida imagen en pixeles
        #tf.keras.layers.Dense(128, activation = "relu"),# Agrega una capa densa que es una de las mas usadas
        #tf.keras.layers.Dense(256, activation = "relu"),
        #tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(10, activation = "softmax")# Agregamos la capa final que en este caso es la capa densa de 10 nodos por que tenemos 10 diferentes salidas
    ])
    # Compilamos red neuronal
    model.compile(
        loss = 'sparse_categorical_crossentropy', # Funcion de perdida
        optimizer = "adam", # Optimizador
        metrics = ['accuracy'] # Metrica que usaremos para medir la eficacion del modelo
    )

    callback = Mycallback()

    history = model.fit(
        x_train,
        y_train,
        validation_data = (x_test, y_test),
        epochs = 100, # Epocas
        callbacks = [callback]

    )
    return history.epoch, history.history['accuracy'][-1]

if __name__ == '__main__':
    train_fmnist()
