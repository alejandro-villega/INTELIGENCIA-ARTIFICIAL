# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np

# Datos de entrenamiento (ejemplo)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas
y = np.array([[0], [1], [1], [0]])  # Salidas deseadas

# Definir el modelo de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),  # Capa oculta 1 con 2 neuronas
    tf.keras.layers.Dense(2, activation='sigmoid'),  # Capa oculta 2 con 2 neuronas
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X, y, epochs=10000, verbose=0)  # 10,000 épocas para entrenar

# Evaluar el modelo
loss, accuracy = modelo.evaluate(X, y)
print(f'Pérdida (Loss): {loss}')
print(f'Precisión (Accuracy): {accuracy}')

# Hacer predicciones
predicciones = modelo.predict(X)
print('Predicciones:')
for i in range(len(X)):
    print(f'Entrada: {X[i]}, Salida Deseada: {y[i]}, Predicción: {predicciones[i]}')
