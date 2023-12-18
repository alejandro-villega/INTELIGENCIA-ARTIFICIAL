# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:02:14 2023

@author: AlejandroVillega
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

@tf.autograph.experimental.do_not_convert
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Capa personalizada que implementa la función simulador
class SimuladorLayer(keras.layers.Layer):
    def __init__(self, parametros_iniciales, **kwargs):
        super(SimuladorLayer, self).__init__(**kwargs)
        self.parametros_ajustados = self.add_weight(
            shape=(5,), 
            initializer=tf.constant_initializer(parametros_iniciales), 
            trainable=True
        )

    def call(self, inputs):
        a = self.parametros_ajustados[0]
        b = self.parametros_ajustados[1]
        c = self.parametros_ajustados[2]
        d = self.parametros_ajustados[3]
        e = self.parametros_ajustados[4]
        return a * (inputs ** 4) + b * (inputs ** 3) + c * (inputs ** 2) + d * inputs + e

# Datos de entrada y resultados de ensayos
X_simulador = np.load('datos_simulador.npy')
y_ensayos = np.load('resultados_ensayos.npy')

# Valores iniciales
parametros_ajustados = np.array([2.0, 1.5, 1.0, 0.5, 0.2], dtype=np.float32)

# Convertir los datos de entrenamiento a tensores TensorFlow
X_entrenamiento = tf.constant(X_simulador, dtype=tf.float32)
y_entrenamiento = tf.constant(y_ensayos, dtype=tf.float32)

# Función de pérdida personalizada
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Construir el modelo con la capa personalizada
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)  # 5 neuronas en la capa de salida
])

# Agregar la capa personalizada después de la última capa Dense
model.add(SimuladorLayer(parametros_ajustados))

# Compilar el modelo utilizando la función de pérdida personalizada y el optimizador
#optimizer = keras.optimizers.Adam()  # Puedes ajustar los parámetros del optimizador según sea necesario
optimizer = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=custom_loss)

# Número de epochs
num_epochs = 300

# Bucle de entrenamiento
for epoch in range(num_epochs):
    history = model.fit(X_entrenamiento, y_entrenamiento, epochs=1, verbose=0)

    # Obtener los resultados de la última capa densa para el último dato de entrenamiento
    resultados_densa = model.predict(X_entrenamiento)
    parametros_ajustados = resultados_densa[-1]
    print("nuevos_parametros",parametros_ajustados)
    # Actualizar los parámetros ajustados de la capa personalizada
    model.layers[-1].parametros_ajustados.assign(parametros_ajustados)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {history.history["loss"][0]}')

# ...

# Visualización de las predicciones y datos no normalizados
X_visualizacion = np.linspace(0, 10, 100)  # Rango no normalizado para visualización
y_visualizacion = model.predict(X_visualizacion)

plt.figure(figsize=(8, 6))
plt.scatter(X_simulador, y_ensayos, label='Datos de entrenamiento (no normalizados)', alpha=0.5)
plt.plot(X_visualizacion, y_visualizacion, 'r', label='Predicciones')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste del modelo a la función simulada')
plt.legend()
plt.show()
