# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 07:42:25 2023

@author: AlejandroVillega
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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

# Valores iniciales
parametros_ajustados = np.array([2.0, 1.5, 1.0, 0.5, 0.2], dtype=np.float32)

# Simulador de datos
def simulador(x, parametros_ajustados):
    a, b, c, d, e = parametros_ajustados
    return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e

# Crear datos de entrenamiento
X_entrenamiento = np.linspace(0, 10, 100)  # Rango no normalizado
y_entrenamiento = [simulador(x, parametros_ajustados) for x in X_entrenamiento]

# Convertir los datos de entrenamiento a tensores TensorFlow
X_entrenamiento = tf.constant(X_entrenamiento, dtype=tf.float32)
y_entrenamiento = tf.constant(y_entrenamiento, dtype=tf.float32)

# Construir el modelo con la capa personalizada
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)  # 5 neuronas en la capa de salida
])

# Agregar la capa personalizada después de la última capa Dense
model.add(SimuladorLayer(parametros_ajustados))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Obtener la función para obtener la salida de la última capa densa
get_densa_output = tf.keras.backend.function([model.input], [model.layers[-2].output])

# Número de epochs
num_epochs = 100
print("parametros_ajustados",parametros_ajustados)
# ... (código previo)

# Bucle de entrenamiento
for epoch in range(num_epochs):
    # Entrenar por un epoch
    history = model.fit(X_entrenamiento, y_entrenamiento, epochs=1, verbose=0)

    # Obtener los resultados de la última capa densa para el último dato de entrenamiento
    resultado_ultimo_dato = model.predict(X_entrenamiento[-1:])
    nuevos_parametros = resultado_ultimo_dato[0]
    print("nuevos_parametros",nuevos_parametros)
    # Actualizar los parámetros ajustados de la capa personalizada
    model.layers[-1].parametros_ajustados.assign(nuevos_parametros)

    # Imprimir algunos resultados o realizar otras operaciones si es necesario
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {history.history["loss"][0]}')

# ... (resto del código)


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
