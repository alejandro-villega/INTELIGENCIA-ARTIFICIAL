# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:02:02 2023

@author: AlejandroVillega
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
historia = modelo.fit(X, y, epochs=10000, verbose=0)  # 10,000 épocas para entrenar

# Evaluar el modelo
loss, accuracy = modelo.evaluate(X, y)
print(f'Pérdida (Loss): {loss}')
print(f'Precisión (Accuracy): {accuracy}')

# Hacer predicciones
predicciones = modelo.predict(X)
print('Predicciones:')
for i in range(len(X)):
    print(f'Entrada: {X[i]}, Salida Deseada: {y[i]}, Predicción: {predicciones[i]}')

# Graficar los datos de entrada y las predicciones
plt.figure(figsize=(10, 5))

# Gráfico de los datos de entrada
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis')
plt.title('Datos de Entrada')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Gráfico de las predicciones
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predicciones.flatten(), cmap='viridis')
plt.title('Predicciones')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()