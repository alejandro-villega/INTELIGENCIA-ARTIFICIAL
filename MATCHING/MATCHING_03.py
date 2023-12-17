# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:53:27 2023

@author: AlejandroVillega
"""
import tensorflow as tf
import numpy as np

# Función de simulador con parámetros
class SimuladorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimuladorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Definir los parámetros del simulador como pesos entrenables
        self.a = self.add_weight("a", shape=(1,), initializer="ones", trainable=True)
        self.b = self.add_weight("b", shape=(1,), initializer="ones", trainable=True)
        self.c = self.add_weight("c", shape=(1,), initializer="ones", trainable=True)
        self.d = self.add_weight("d", shape=(1,), initializer="ones", trainable=True)
        self.e = self.add_weight("e", shape=(1,), initializer="ones", trainable=True)
        super(SimuladorLayer, self).build(input_shape)

    def call(self, x):
        # Calcular la salida del simulador
        return self.a * (x ** 4) + self.b * (x ** 3) + self.c * (x ** 2) + self.d * x + self.e

# Datos de entrada para el entrenamiento
X_simulador = np.random.rand(100, 1)  # Datos de entrada
resultados_reales = 3 * (X_simulador ** 4) - 2 * (X_simulador ** 3) + 0.5 * (X_simulador ** 2) + 1.5 * X_simulador - 2.5  # Resultados reales

# Definir la arquitectura de la red neuronal
inputs = tf.keras.layers.Input(shape=(1,))
simulador = SimuladorLayer()(inputs)
output = tf.keras.layers.Dense(1, activation='linear')(simulador)

model = tf.keras.models.Model(inputs=inputs, outputs=output)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar la red neuronal
model.fit(X_simulador, resultados_reales, epochs=1000, verbose=1)

# Obtener los parámetros ajustados
parametros_ajustados = model.get_layer('simulador_layer').get_weights()

# Función para realizar predicciones usando los parámetros ajustados
def simulador_ajustado(X):
    a, b, c, d, e = parametros_ajustados
    return a * (X ** 4) + b * (X ** 3) + c * (X ** 2) + d * X + e

# Generar datos de prueba para hacer predicciones
X_prueba = np.linspace(0, 1, 100)
predicciones = [simulador_ajustado(x) for x in X_prueba]

# Comparar predicciones con resultados reales
import matplotlib.pyplot as plt
plt.plot(X_prueba, predicciones, label='Predicciones')
plt.plot(X_simulador, resultados_reales, 'ro', label='Datos de entrenamiento')
plt.legend()
plt.show()
