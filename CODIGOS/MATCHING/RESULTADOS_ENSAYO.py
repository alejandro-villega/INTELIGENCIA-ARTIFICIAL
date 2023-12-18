# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:30:52 2023

DATOS DE ENSAYO

@author: AlejandroVillega
"""

import numpy as np

# Supongamos que tienes datos de entrada y resultados de ensayos en arreglos NumPy
datos_simulador = np.array([1,	2	,3	,4	,5,	6,	7,	8,	9,	10])  # Datos de entrada
resultados_ensayos = np.array([6,	12,	15,	15,	13,	1,	-1,	-1,	4,	10])  # Resultados de ensayos

# Guarda los arreglos en archivos NumPy
np.save('datos_simulador.npy', datos_simulador)
np.save('resultados_ensayos.npy', resultados_ensayos)