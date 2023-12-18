# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:54:55 2023

@author: AlejandroVillega
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar el modelo Entrenado desde el archivo
modelo_cargado = load_model('modelo_entrenado_GPT_05.h5')


# Generar datos de entrada y salidas deseadas basadas en la función
x1=	0.3
x2=	1.5
x3=	2.325
x4=	3.60375
x5=	5.5858125


X = np.arange(0, 3.7, 3.7/1000)  # Valores de entrada de 0 a 1000 

y = (X-x1)*(X-x2)*(X-x3)*(X-x4)*(X-x5)  # Salidas deseadas basadas en el seno de los valores
# Normalizar los datos de entrada y salidas deseadas entre 0 y 1
X_normalized = X /3.7
y_normalized = y/20    # Escalar la función  al rango [0, 1]

# Redimensionar los datos de entrada para que sean bidimensionales
X_normalized = X_normalized.reshape(-1, 1)



# Realizar una predicción con el modelo cargado
predicciones_normalized = modelo_cargado.predict(X_normalized)

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