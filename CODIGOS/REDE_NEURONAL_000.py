# -*- coding: utf-8 -*-
"""
Creando las capas de neuronas
Para poder programar una capa de neuronas, primero debemos entender bien cómo funcionan. Básicamente una 
red neuronal funciona de la siguiente manera:

1-Una capa recibe valores, llamados inputs. En la primera capa, esos valores vendrán definidos por los 
datos de entrada, mientras que el resto de capas recibirán el resultado de la capa anterior.
2-Se realiza una suma ponderada todos los valores de entrada. Para hacer esa ponderación necesitamos una 
matriz de pesos, conocida como W. La matriz W tiene tantas filas como neuronas la capa anterior y tantas 
columnas como neuronas tiene esa capa.
3-Al resultado de la suma ponderada anterior se le sumará otro parámetro, conocido como bias o, 
simplemente, b. 
En este caso, cada neurona tiene su propio bias, por lo que las dimensiones del vector bias será una 
columna y tantas filas como neuronas tiene esa capa.
4-Por cuarto lugar tenemos una de las claves de las redes neuronales: la función de activación. 
Y es que, si te das cuenta, lo que tenemos hasta ahora no es más que una regresión lineal. 
Para evitar que toda la red neuronal se pueda reducir a una simple regresión lineal, al resultado de la 
suma del bias a la suma ponderada se le aplica una función, conocido como función de activación. 
El resultado de esta función será el resultado de la neurona.
Por tanto, para poder montar una capa de una red neuronal solo necesitamos saber el número de neuronas 
en la capa y el número de neuronas de la capa anterior. Con eso, podremos crear tanto W como b


Para crear esta estructura vamos a crear una clase, que llamaremos capa. Además, vamos a inicializar 
los parámetros (b y W) con datos aleatorios. 
Para esto último usaremos la función trunconorm de la librería stats, ya que nos permite crear 
datos aleatorios dado un rango, media y desviación estándar, lo cual hará que a nuestra red le cueste 
menos arrancar. In [1]:
"""

from scipy import stats

class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)
    
"""
Funciones de Activación

Como he dicho antes, al resultado de la suma ponderada del input y el parámetro bias se aplica una 
función de activación, es decir, una transformación a los datos. El motivo es que, si no lo hiciéramos, 
cada neurona lo único que haría sería una transformación lineal de los datos, dando como resultado una 
función lineal normal.

¿Qué función usamos? Podríamos usar cualquier función de activación que haga que el resultado no sea 
lineal, pero generalmente se suelen usar dos: función sigmoide y función ReLu

Función de activación: Función Sigmoide
La función sigmoide básicamente recibe un valor x y devuelve un valor entre 0 y 1. 
Esto hace que sea una función muy interesante, ya que indica la probabilidad de un estado. 
Por ejemplo, si usamos la función sigmoide en la última capa para un problema de clasificación 
entre dos clases, la función devolverá la probabilidad de pertenencia a un grupo. In [2]

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
La función ReLu es muy simple: para valores negativos, la función devuelve cero.
Para valores positivos, la función devuelve el mismo valor. 
Pero, a pesar de ser tan simple, esta función es la función de activación más usada en el campo de las redes neuronales 
y deep learning. ¿El motivo? Pues precisamente porque es sencilla y porque evita el gradient vanish (más info aquí). 
In [3]:
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

Con eso y con lo que hemos programado hasta ahora ya podemos crear la estructura de nuestra red neuronal.


En nuestro caso, usaremos la red neuronal para solucionar un problema de clasificación de dos clases, 
para lo cual usaremos una red pequeña, de 4 capas que se compondrá de:

Una capa de entrada con dos neuronas, ya que usaremos dos variables.
Dos capas ocultas, una de 4 neuronas y otra de 8.
Una capa de salida, con una única neurona que predecirá la clase

Asimismo, tenemos que definir qué función de activación se usará en cada capa. En nuestro caso, 
usaremos la función ReLu en todas las capas menos en la última, en la cual usaremos la función sigmoide. 
Es importante recordar que en la primera capa solo se reciben los datos, no se aplica una función ni nada.


Por otro lado, Python no permite crear una lista de funciones. 
Por eso, hemos definido las funciones relu y sigmoid como funciones ocultas usando lambda. In [4]:
"""

# Numero de neuronas en cada capa. 
# El primer valor es el numero de columnas de la capa de entrada.
neuronas = [2,4,8,1] 

# Funciones de activacion usadas en cada capa. 
funciones_activacion = [relu,relu, sigmoid]

"""
Con todo esto, ya podemos crear la estructura de nuestra red neuronal programada en Python.
Lo haremos de forma iterativa e iremos guardando esta estructura en un nuevo objeto,
llamado red_neuronal. In [5]
"""

red_neuronal = []

for paso in range(len(neuronas)-1):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x)

print(red_neuronal)


"""
Con esto ya tenemos la estructura de nuestra red neuronal. 
Ahora solo quedarían dos pasos más: por un lado, conectar la red para que nos de una predicción y 
un error y, por el otro lado, ir propagando ese error hacia atrás para ir entrenando a nuestra red 
neuronal.

Haciendo que nuestra red neuronal prediga
Para que nuestra red neuronal prediga lo único que tenemos que hacer es definir los cáculos que tiene 
que seguir. 
Como he comentado anterirormente, son 3 los cálculos a seguir: 
    
    multiplicar los valores de entrada por la matriz de pesos W y 
    sumar el parámetro bias (b) y 
    aplicar la función de activación.

Para multiplicar los valores de entrada por la matriz de pesos tenemos que hacer una multiplicación matricial. 
Veamos el ejemplo de la primera capa: In [6]:
"""

X =  np.round(np.random.randn(20,2),3) # Ejemplo de vector de entrada

z = X @ red_neuronal[0].W

print(z[:10,:], X.shape, z.shape)

"""
Ahora, hay que sumar el parámetro bias (b) al resultado anterior de z. In [7]
"""
z = z + red_neuronal[0].b

print(z[:5,:])

"""
Ahora, habría que aplicar la función de activación de esa capa. In [8]
"""
a = red_neuronal[0].funcion_act[0](z)
a[:5,:]

"""
Con esto, tendríamos el resultado de la primera capa, que a su vez es la entrada para la
 segunda capa y así hasta la última. 
 Por tanto, queda bastante claro que todo esto lo podemos definir de forma iterativa 
 dentro de un bucle. In [9]:
"""

output = [X]

for num_capa in range(len(red_neuronal)):
  z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
  a = red_neuronal[num_capa].funcion_act[0](z)
  output.append(a)

print(output[-1])


"""
Así, tendríamos la estimación para cada una de las clases de este ejercicio de prueba. 
Como es la primera ronda, la red no ha entrenado nada, por lo que el resultado es aleatorio.
Por tanto, solo quedaría una cosa: entrenar a nuestra red neuronal programada en Python. 
¡vamos a ello!


Entrenar tu red neuronal
Creando la función de coste

Para poder entrenar la red neuronal lo primero que debemos hacer es calcular cuánto ha fallado. 
Para ello usaremos uno de los estimadores más típicos en el mundo del machine learning: 
    el error cuadrático medio (MSE).

Calcular el error cuadrático medio es algo bastante simple: a cada valor predicho le restas
el valor real, lo elevas al cuadrado, haces la suma ponderada y calculas su raíz.
Además, como hemos hecho anteriormente aprovecharemos para que esta misma función nos devuelva 
la derivada dela función de coste, la cual nos será útil en el paso de backpropagation. In [10]:
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

from random import shuffle

Y = [0] * 10 + [1] * 10
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)

mse(output[-1], Y)[0]


"""
Ahora que ya tenemos el error calculado, tenemos que irlo propagando hacia atrás 
para ir ajustando los parámetros. Haciendo esto de forma iterativa, nuestra red 
neuronal irá mejorando sus predicciones, es decir, disminuirá su error. Vamos, 
que así es como se entrena a una red neuronal

Backpropagation y gradient descent: entrenando a nuestra red neuronal

Gradient descent: optimizando los parámetros
Con el algoritmo de gradient descent optimizaremos los parámetros para así ir 
mejorando los resultados de nuestra red. Si volvemos atrás, los parámetros los 
hemos inicializado de forma aleatoria. Por eso, eso poco probable que sus 
valores sean los mejores para nuestra red neuronal. Supongamos, por ejemplo, 
que nuestros parámetros se han inicializado en esta posición.

Como véis, los valores están lejos del valor óptimo (el azul oscuro más abajo), 
por lo que deberíamos hacer que nuestro parámetro llegue a allí. 
Pero, ¿cómo lo hacemos?

Para ello, usaremos gradient descent. Este algoritmo utiliza el error en el punto 
en el que nos encontramos y calcula las derivadas parciales en dicho punto. 
Esto nos devuelve el vector gradiente, es decir, un vector de direcciones hacia 
donde el error se incrementa. Por tanto, si usamos el inverso de ese valor, 
iremos hacia abajo. 
En definitiva, gradient descent calcula la inversa del gradiente para saber qué 
valores deben tomar los hiperparámetros.

Cuánto nos movamos hacia abajo dependerá de otro hiperparámetro: el learning rate. 
Este hiperparámetro no se suele optimizar, aunque si que hay que tener en cuenta dos cuestiones:

1-Si el valor del learning rate es muy bajo, el algoritmo tardará en aprender, 
porque cada paso será muy corto.
2-Si el learning rate es muy grande, puede que te pases del valor óptimo, 
por lo que no llegues a encontrar el valor óptimo de los parámetros.

Para evitar esto se pueden aplicar varias técnicas, como la de disminuir el 
learning rate a cada paso que demos, por ejemplo. En nuestro caso, no nos vamos 
a complicar y dejaremos un learning rate fijo.

Con gradient descent a cada iteración nuestros parámetros se irán acercando a un valor óptimo,
hasta que lleguen a un punto óptimo, a partir del cual nuestra red dejará de aprender.

Esto suena muy bien, pero como he dicho, gradient descent utiliza el error en el punto. 
Este error ya lo tenemos para nuestro vector de salida, pero, ¿que pasa en el resto de capas? 
Para eso, usamos backpropagation.

Backpropagation: calculando el error en cada capa

En nuestra red neuronal todos los pasos previos a la neurona de salida tendrán un impacto en el mismo:

    el error de la primera capa influirá en el error de la segunda capa, los de la primera y
    segunda influirán en los de la tercera y así sucesivamente.
    
Por tanto, la única manera de calcular el error de cada neurona en cada capa es haciendo 
el proceso inverso: primero calculamos el error de la última capa, con lo que podremos 
calcular el error de la capa anterior y así hasta completar todo el proceso.

Además, este es un proceso eficiente, ya que podemos aprovechar la propagación hacia 
atrás para ir ajustando los parámetros W y b mediante gradient descent. 
En cualquier caso, para calcular el descenso del gradiente necesitamos aplicar derivadas, 
entre las que se encuentra las derivadas de la función de coste. Por eso mismo, 
al definir las funciones de activación hemos definido también sus derivadas, 
ya que eso nos ahorrará mucho el proceso.


Dicho esto, veamos cómo funcionan gradient descent y backpropagation.
Para ello,vamos a ver qué valores tienen inicialmente nuestros parámetros W y b en una capa cualquiera,
como por ejemplo la última. In [12]:

"""
red_neuronal[-1].b
red_neuronal[-1].W


"""
Como desconocemos el valor óptimo de estos parámetros, los hemos inicializado de forma aleatoria. 
Por tanto, en cada ronda estos valores se irán cambiando pooco a poco. Para ello, 
lo primero que debemos hacer es transmitir el error hacia atrás. 
Como estamos trabajando de atrás hacia adelante (o de derecha a izquierda si visualizamos la red), 
partiremos de la última capa e iremos hacia adelante.

El error lo calculamos como la derivada de la función de coste sobre el resultado 
de la capa siguiente por la derivada de la función de activación. 
En nuestro caso, el resultado del último valor está en la capa -1, mientras que 
la capa que vamos a optimizar es la anteúltima (posición -2). Además, como hemos 
definido las funciones como un par de funciones, simplemente tendremos que indicar 
el resultado de la función en la posición [1] en ambos casos. In [13]:
"""
# Backprop en la ultima capa
a = output[-1]
x = mse(a,Y)[1] * red_neuronal[-2].funcion_act[1](a)

print(x)


"""
Si hiciéramos esto en cada capa, iríamos propagando el error generado por la 
estimación de la red neuronal. Sin embargo, propagar el error por si mismo no hace nada, 
sino que ahora tenemos que usar ese error para optimizar los valores de los 
parámetros mediante gradient descent. 
Para ello, tenemos calcular las derivadas en el punto de los parámetros b y W y 
restar esos valores a los valores anteriores de b y W. In [14]:
"""

red_neuronal[-1].b = red_neuronal[-1].b - x.mean() * 0.01
red_neuronal[-1].W = red_neuronal[-1].W - (output[-1].T @ x) * 0.01

print(red_neuronal[-1].b)
print(red_neuronal[-1].W)


"""
Con esto ya habríamos actualizado los parámetros de W y b en la última capa. 
Ahora bien, para calcular el error de la siguiente capa tendríamos que multiplicar 
matricialmente el error de esta capa (x) por los pesos de la misma, para así 
saber cuánto de ese error corresponde a cada neurona de la capa. 
Pero claro, ya hemos actualizado los pesos, por lo que eso fastidiaría el aprendizaje, ¿no?

Efectivamente, eso nos generaría un problema y tendríamos que esperar una 
iteración más para aplicar cambios. Sin embargo, tiene solución y muy fácil. 
Para evitar ese problema lo que hacemos es guardar los valores de W antes de 
actualizar en una variable «temporal», que en mi caso he llamado W_temp. 
De esta manera, somos capaces de calcular el error correspondiente a cada neurona 
y actualizar los valores de los parámetros todo en una misma iteración.

Si ponemos todo esto junto, la fórmula de backpropagation y gradient descent 
queda de la siguiente manera: In [15]:
"""

# Definimos el learning rate
lr = 0.05

# Creamos el indice inverso para ir de derecha a izquierda
back = list(range(len(output)-1))
back.reverse()

# Creamos el vector delta donde meteremos los errores en cada capa
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


print('MSE: ' + str(mse(output[-1],Y)[0]) )
print('Estimacion: ' + str(output[-1]) )