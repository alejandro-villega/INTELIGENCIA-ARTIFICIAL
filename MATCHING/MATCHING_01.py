import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Datos de entrada y resultados de ensayos
X_simulador = np.load('datos_simulador.npy')
y_ensayos = np.load('resultados_ensayos.npy')




# Función de simulador con coeficientes
def simulador(X, coeficientes):
    grado = len(coeficientes) - 1
    Y = sum(coeficientes[i] * (X ** (grado - i)) for i in range(grado + 1))
    return Y

# Modelo de red neuronal
modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Dense(7, activation='linear', input_dim=1))
# Agregar una capa para aprender los coeficientes
modelo.add(tf.keras.layers.Dense(5, activation='linear', name='coeficientes'))
# Capa de Salida
modelo.add(tf.keras.layers.Dense(1))

# Función de pérdida personalizada que utiliza el simulador
def custom_loss(y_true, y_pred, coeficientes):
    Y_simulado = simulador(X_simulador, coeficientes)
    return tf.reduce_mean(tf.square(Y_simulado - y_pred))

# Coeficientes iniciales (puedes inicializarlos de la manera que desees)
coeficientes_iniciales = [0., 1.0,1.0, 1.0, 1.0]

# Compilar el modelo con la pérdida de error cuadrático medio y la función de pérdida personalizada
modelo.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, coeficientes_iniciales))

# Inicializar una lista para almacenar las pérdidas
losses = []

# Función para actualizar el gráfico en tiempo real
def update_plot(epoch, logs):
    losses.append(logs['loss'])
    plt.cla()
    plt.plot(losses)
    plt.title('Evolución de Pérdidas')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.pause(0.1)

# Configurar una devolución de llamada para actualizar el gráfico
plot_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=update_plot)

# Entrenar el modelo
modelo.fit(X_simulador, y_ensayos, epochs=10, verbose=0, callbacks=[plot_callback])

# Obtener las predicciones de los coeficientes calibrados
predicciones_coeficientes = modelo.predict(X_simulador)

# Los coeficientes calibrados son los valores de predicciones_coeficientes
coeficientes_calibrados = predicciones_coeficientes

# Ahora coeficientes_calibrados contiene los valores de los coeficientes calibrados
print("Coeficientes calibrados:", coeficientes_calibrados)


# Mostrar el gráfico final
plt.show()