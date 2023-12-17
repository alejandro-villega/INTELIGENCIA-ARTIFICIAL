# -*- coding: utf-8 -*-
"""
Creando las capas de neuronas

"""

from scipy import stats

class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)
    
"""
Funciones de Activación



"""
import numpy as np
import math
import matplotlib.pyplot as plt


sigmoid = (
  lambda x:1 / (1 + np.exp(-x)),
  lambda x:x * (1 - x)
  )

rango = np.linspace(-10,10).reshape([50,1])
datos_sigmoide = sigmoid[0](rango)
datos_sigmoide_derivada = sigmoid[1](rango)

#Cremos los graficos
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)
fig.tight_layout()

"""
Función de activación: Función ReLu

"""
def derivada_relu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

relu = (
  lambda x: x * (x > 0),
  lambda x:derivada_relu(x)
  )

datos_relu = relu[0](rango)
datos_relu_derivada = relu[1](rango)


# Volvemos a definir rango que ha sido cambiado
rango = np.linspace(-10,10).reshape([50,1])

# Cremos los graficos
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
axes[0].plot(rango, datos_relu[:,0])
axes[1].plot(rango, datos_relu_derivada[:,0])
plt.show()

"""
Programando una red neuronal en Python
Para crear una red neuronal, simplemente tendremos que indicar tres cosas: 
    el número de capas que tiene la red, 
    el número de neuronas en cada capa y 
    la función de activación que se usará en cada una de las capas. 

"""

# Numero de neuronas en cada capa. 
# El primer valor es el numero de columnas de la capa de entrada.
neuronas = [1,1] 

# Funciones de activacion usadas en cada capa. 
# funciones_activacion = [relu, sigmoid]
funciones_activacion = [relu]


red_neuronal = []

for paso in range(len(neuronas)-1):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x)

print(red_neuronal)
'''
# Haciendo que nuestra red neuronal prediga

'''

X = ([0],
[100])


Y =([5],
[405])



output = [np.array(X)]
print("b = ",red_neuronal[-1].b)
print("W = ",red_neuronal[-1].W) 

# Haciendo que nuestra red neuronal prediga
for num_capa in range(len(red_neuronal)):
  z = output[num_capa] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
  a = red_neuronal[num_capa].funcion_act[0](z)
  output.append(a)

print(output[-1])


"""
Entrenar tu red neuronal
Creando la función de coste
"""

def mse(Ypredich, Yreal):

  # Calculamos el error
  x = (np.array(Ypredich) - np.array(Yreal)) ** 2
  x = np.mean(x)

  # Calculamos la derivada de la funcion
  y = np.array(Ypredich) - np.array(Yreal)
  return (x,y)

"""
Con esto, vamos a «inventarnos» unas clases (0 o 1) para los valores que 
nuestra red neuronal ha predicho antes. Así, calcularemos el error cuadrático medio. In [11]:
"""


mse(output[-1], Y)[0]


"""

Backpropagation y gradient descent: entrenando a nuestra red neuronal

Gradient descent: optimizando los parámetros


"""


# Definimos el learning rate
lr = 0.05

# Creamos el indice inverso para ir de derecha a izquierda
back = list(range(len(output)-1))
back.reverse()

# Creamos el vector delta donde meteremos los errores en cada capa


values = range(10)
    
for iteracion in values:
    delta = []   
    for capa in back:
      # Backprop #
    
      # Guardamos los resultados de la ultima capa antes de usar backprop para poder usarlas en gradient descent
      a = output[capa+1][1]
    
      # Backprop en la ultima capa 
      if capa == back[0]:
        x = mse(a,Y)[1] * red_neuronal[capa].funcion_act[1](a)
        delta.append(x)
    
      # Backprop en el resto de capas 
      else:
        x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
        delta.append(x)
    
      # Guardamos los valores de W para poder usarlos en la iteracion siguiente
      W_temp = red_neuronal[capa].W.transpose()
    
      # Gradient Descent #
    
      # Ajustamos los valores de los parametros de la capa
      red_neuronal[capa].b = red_neuronal[capa].b - delta[-1].mean() * lr
      red_neuronal[capa].W = red_neuronal[capa].W - (output[capa].T @ delta[-1]) * lr
    print("iteracion = ",iteracion)
    print('MSE: ' + str(mse(output[-1],Y)[0]) )
    print('Estimacion: ', (output[-1]) )    
    print("b = ",red_neuronal[-1].b)
    print("W = ",red_neuronal[-1].W) 
   
# Haciendo que nuestra red neuronal prediga
    for num_capa in range(len(red_neuronal)):
      z = output[num_capa] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
      a = red_neuronal[num_capa].funcion_act[0](z)
      output.append(a)
    
#    print(output[-1])

print("iteracion = ",iteracion)
print('MSE: ' + str(mse(output[-1],Y)[0]) )
print('Estimacion: ', (output[-1]) )
#    print("b = ",red_neuronal[-1].b)
#    print("W = ",red_neuronal[-1].W)
