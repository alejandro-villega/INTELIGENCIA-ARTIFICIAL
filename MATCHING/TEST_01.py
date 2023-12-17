# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 08:58:57 2023

@author: AlejandroVillega
"""

import tensorflow as tf

# Crea un modelo
modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Input(shape=(1,)))
modelo.add(tf.keras.layers.Dense(64, activation='relu'))
modelo.add(tf.keras.layers.Dense(32, activation='relu', name='mi_capa'))
modelo.add(tf.keras.layers.Dense(1, activation='linear'))

# Carga pesos entrenados si es necesario
# modelo.load_weights('pesos_entrenados.h5')

# Define los datos de entrada
X = [[1.0]]

# Utiliza predict_on_batch para obtener las salidas de "mi_capa"
resultados = modelo.predict_on_batch(X)

# "resultados" contendrá las salidas de la capa "mi_capa" para la entrada "X"
print(resultados)
