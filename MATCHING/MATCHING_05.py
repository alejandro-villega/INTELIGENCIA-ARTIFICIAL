# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:38:53 2023

@author: AlejandroVillega
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Simulador de datos
def simulador(x):
    a, b, c, d, e = 2.0, 1.5, 1.0, 0.5, 0.2
    return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e

# Crear datos de entrenamiento
X_entrenamiento = np.linspace(0, 10, 100)  # Rango no normalizado
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
history = model.fit(X_entrenamiento, y_entrenamiento, epochs=1000, verbose=0)

# Probar el modelo ajustado
X_prueba = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)  # Prueba con valores no normalizados
predicciones = model.predict(X_prueba)

# Imprimir las predicciones
for x, pred in zip(X_prueba, predicciones):
    print(f'Entrada: {x}, Parámetros ajustados: {pred}, Valor real: {simulador(x)}')

# Visualización de las predicciones y datos no normalizados
X_visualizacion = np.linspace(0, 10, 100)  # Rango no normalizado para visualización
y_visualizacion = model.predict(X_visualizacion)

plt.figure(figsize=(8, 6))
plt.scatter(X_entrenamiento, y_entrenamiento, label='Datos de entrenamiento (no normalizados)', alpha=0.5)
plt.plot(X_visualizacion, y_visualizacion, 'r', label='Predicciones')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste del modelo a la función simulada')
plt.legend()
plt.show()
