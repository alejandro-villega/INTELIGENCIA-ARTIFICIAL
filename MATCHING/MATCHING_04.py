# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:53:27 2023

@author: AlejandroVillega
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Simulador de datos
def simulador(x):
    a, b, c, d, e = 2.0, 1.5, 1.0, 0.5, 0.2
    return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e

# Crear datos de entrenamiento
X_entrenamiento = np.linspace(0, 1, 100)
y_entrenamiento = [simulador(x) for x in X_entrenamiento]

# Convertir los datos de entrenamiento a tensores TensorFlow
X_entrenamiento = tf.constant(X_entrenamiento, dtype=tf.float32)
y_entrenamiento = tf.constant(y_entrenamiento, dtype=tf.float32)

# Construir el modelo
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)  # 5 neuronas en la capa de salida
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_entrenamiento, y_entrenamiento, epochs=1000, verbose=1)

# Obtener los parámetros ajustados
parametros_ajustados = model.layers[-1].get_weights()[0]

# Probar el modelo ajustado
X_prueba = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
predicciones = model.predict(X_prueba)

# Imprimir las predicciones
for x, pred in zip(X_prueba, predicciones):
    print(f'Entrada: {x}, Parámetros ajustados: {pred}, Valor real: {simulador(x)}')
