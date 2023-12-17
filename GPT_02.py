# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:06:29 2023

@author: AlejandroVillega
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de entrada y salidas deseadas basadas en la función seno
X = np.arange(0, 1001, 1)  # Valores de entrada de 0 a 1000 en grados
y = np.sin(np.radians(X))  # Salidas deseadas basadas en el seno de los valores

# Normalizar los datos de entrada y salidas deseadas entre 0 y 1
X_normalized = X / 1000.0
y_normalized = (y + 1) / 2.0  # Escalar la función seno al rango [0, 1]

# Definir el modelo de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),  # Capa oculta 1 con 8 neuronas y ReLU
    tf.keras.layers.Dense(1, activation='linear')  # Capa de salida con activación lineal
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
historia = modelo.fit(X_normalized, y_normalized, epochs=1000, verbose=0)

# Evaluar el modelo
loss = modelo.evaluate(X_normalized, y_normalized)
print(f'Pérdida (Loss): {loss}')

# Hacer predicciones
predicciones_normalized = modelo.predict(X_normalized)
predicciones = (predicciones_normalized * 2) - 1  # Deshacer la escala para obtener predicciones en el rango [-1, 1]

# Graficar los datos de entrada, las salidas deseadas y las predicciones
plt.figure(figsize=(10, 5))

# Gráfico de la función seno deseada y las predicciones
plt.plot(X_normalized, y_normalized, label='Función Seno Deseada', linestyle='--', color='blue')
plt.plot(X_normalized, predicciones_normalized, label='Predicciones', color='red')
plt.title('Función Seno Deseada vs. Predicciones')
plt.xlabel('Valor de Entrada (Normalizado)')
plt.ylabel('Salida (Normalizada)')

plt.legend()
plt.show()