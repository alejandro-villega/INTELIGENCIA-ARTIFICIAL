# -*- coding: utf-8 -*-
"""
   FUNSION POLINOMIO

Created on Sat Oct  7 15:11:48 2023

@author: AlejandroVillega
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def build_model(num_layers, num_neurons):
    model = tf.keras.Sequential()
    
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    
    return model

# Generar datos de entrada y salidas deseadas basadas en la función
x1=	0.3
x2=	1.5
x3=	2.325
x4=	3.60375
x5=	5.5858125


X = np.arange(0, 3.7, 3.7/1000)  # Valores de entrada de 0 a 1000 en grados
#y = ao + a1 * X + a2 * X**2 +a3 * X**3 + a4 * X**4  # Salidas deseadas basadas en el seno de los valores
y = (X-x1)*(X-x2)*(X-x3)*(X-x4)*(X-x5)  # Salidas deseadas basadas en el seno de los valores
# Normalizar los datos de entrada y salidas deseadas entre 0 y 1
X_normalized = X /3.7
y_normalized = y/20    # Escalar la función  al rango [0, 1]

# Redimensionar los datos de entrada para que sean bidimensionales
X_normalized = X_normalized.reshape(-1, 1)

# Especificar el número de capas y neuronas por capa
num_layers = 10  # Número de capas ocultas
num_neurons = 8  # Número de neuronas por capa oculta

# Construir el modelo con la función build_model
modelo = build_model(num_layers, num_neurons)

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
historia = modelo.fit(X_normalized, y_normalized, epochs=100, verbose=0)

# Evaluar el modelo
loss = modelo.evaluate(X_normalized, y_normalized)
print(f'Pérdida (Loss): {loss}')

# Guardar el modelo en un archivo
modelo.save('modelo_entrenado_GPT_05.h5')

# Hacer predicciones
predicciones_normalized = modelo.predict(X_normalized)
predicciones = (predicciones_normalized * 2) - 1  # Deshacer la escala para obtener predicciones en el rango [-1, 1]

# Graficar los datos de entrada, las salidas deseadas y las predicciones
plt.figure(figsize=(10, 5))

# Gráfico de la función Parabola deseada y las predicciones
plt.plot(X_normalized, y_normalized, label='Función  Deseada', linestyle='--', color='blue')
plt.plot(X_normalized, predicciones_normalized, label='Predicciones', color='red')
plt.title('Función Parabola Deseada vs. Predicciones')
plt.xlabel('Valor de Entrada (Normalizado)')
plt.ylabel('Salida (Normalizada)')

plt.legend()
plt.show()