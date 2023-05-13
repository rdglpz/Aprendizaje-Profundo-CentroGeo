

# Redes neuronales Recurrentes

### Datos secuenciales

Análogamente a las redes neuronales convolucionales, las redes neuronales recurrentes utilizan y aprovechan la estructura temporal de los datos.

Un ejemplo de este tipo de datos son:
* Series de tiempo. (Datos secuenciales más sencillos). El pronóstico es muy útil en muchos ámbitos de aplicación.
* Son también útiles para reconstruir información temporal. 

* En Uber: Modificar precios con antelación. 
* Optimización de carteras de inversión pronosticando el valor de los activos.
* Predicciones de sistemas dinámicos caóticos como en:
	* Clima
	* Contaminación.
	* Cambio de uso de suelo.
	
* Audio
* Texto
	* Predicción basado en secuencia de palabras para evitar problemas generados por otros enfoques como el deBolsa de palabras. 

En bolsa de palabras se crea una "firma" en cada documento contando las palabras que contiene. No importa el orden se pierde información o contexto temporal valioso.



## Redes neuronales recurrentes.

Son una familia de redes neuronales creadas para procesar secuencias de datos.

Se comparten los parámetros en el tiempo en diferentes partes del modelo.

La compartición de parámetros permite crear modelos generalizables para entrada de datos de dimensión variable.

Por ejemplo.

En las frases:

1) "**Fuí** a **Nepal** en **2009**"
2) "En **2009**, yo **fuí** a **Nepal**

Nos gustaría crear un modelo que identifique la información importante sin importar su posición. 

La idea es utilizar convolución en una secuencia temporal. 

La compartición (reutilización) de parámetros, aplica el mismo kernel convolucional a través del tiempo  en cada paso de tiempo $t$

En el caso de RNN, Cada miembro de la salida es producida usando la misma regla (u operación) de actualización para las salidas previas. Esta formulación da lugar en la reutilización de parámetros a través de un grafo computacional profundo.

Una secuencia puede ser representada por 


$$x_1, x_2,\dots,x_t,\dots, x_{\tau}$$

$t$ puede referirse a un tiempo o a una posición en la cadena.

### Grafo computacional recurrente.

Como ejemplo consideremos esta función de recurrencia que representa a un sistema dinámico.

$$\mathbf{s}^{(t)} = f(\mathbf{s}^{(t-1)};\theta)$$ donde $\mathbf{s}_t$ es el estado del sistema en el tiempo $t$.

![enter image description here](https://i.ibb.co/CMgjktZ/classicalds.png)

Otro ejemplo considerando una señal de control $\mathbf{x^{t}}$


$$\mathbf{s}^{(t)} = f(\mathbf{s}^{(t-1)},\mathbf{x^{t}};\theta)$$ donde $\mathbf{s}_t$ contiene información sobre el pasado. 


Para indicar el estado es la unidad oculta de la red podemos expresarlo similarmente a la Ecuación anterior a la anterior como:

$$\mathbf{h}^{(t)} = f(\mathbf{h}^{(t-1)},\mathbf{x^{t}};\theta)$$ donde $\mathbf{h}_t$ contiene información sobre el pasado del estado interno del sistema. 

Para una tarea de predicción, una red aprende como usar este vector de estado $h$ como una variable que recupera información relevante de la secuencia pasada de entrada hasta el tiempo actual $t$.

Esta $\mathbf{h}$ tiene tamaño fijo y sirve para recuperar inform



![enter image description here](https://i.ibb.co/PGqqmPz/recurrentgraph.png)

![enter image description here](https://i.ibb.co/MVGGwVL/unfold-recurrence.png)

Esto permite generar un solo modelo separado $g^{t}$ para todos los pasos de tiempo. Esto es un modelo simple con reutilización de parámetros que permite generalizar secuencias que no aparecen en el conjunto de entrenamiento.

Ejemplos importantes de redes neuronales  recurrentes

![enter image description here](https://i.ibb.co/Nx7cgqd/103-Recurrent-connections.png)

![enter image description here](https://i.ibb.co/R6VXPHT/104-recurrent-output-connection.png)

![enter image description here](https://i.ibb.co/Cz0bFGJ/105-time-unfolded.png)

Ejemplo de Propagación hacia adelante del Esquema presentado en Fig 10.3. 

Propagación hacia adelante comienza con $\mathbf{h}^{(0)}$. 
de $t=1$ a $t=\tau$ aplicar


![enter image description here](https://i.ibb.co/0QJwfMT/fwproprnn.png)

$\matbf{b,c}$, son vetores de sesgo
$\mathbf{U}$  matriz involucrada para  producir el estado oculto $\mathbf{h}^{(t)}$ 
$\mathbf{V}$,   matriz que conecta el estado oculto con la salida 
$\mathbf{W}$ , matriz que conecta el estado oculto previo $\mathbf{h}^{(t-1)}$ con el estado oculto nuevo $\mathbf{h}^{(t)}$  

son matrices que conectan las entradas con el estado oculto, el estado oculto previo con la salida y el estado oculto con el siguiente estado oculto.

### Programando una red neuronal recurrente con Python con ayuda de numpy.

Problema, aprender a predecir una secuencia senoidal con una red neuronal recurrente con el modelo de la Fig 10.3.

$a^{t} = \mathbf{Wh}^{t-1}+ \mathbf{Ux}\mathbf{t}$

$\mathbf{h}^t = \sigma(\mathbf{a}^{(t)})$

$\mathbf{o}^{t} = \mathbf{V}\mathbf{h}^{(t)}$

$\hat{\mathbf{y}}^t =   \mathbf{o}^{t}$ 

$E(\hat{\mathbf{y}}^t,\mathbf{y}^t) = 1/2(\hat{\mathbf{y}}^t - \mathbf{y}^t)^2$



Simplificamos el modelo descrito removiendo los sesgos y 

Recursos 
https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn#how_does_recurrent_neural_networks_work
https://www.youtube.com/watch?v=LHXXI4-IEns
https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-the-vanishing-gradient-problem

















> Written with [StackEdit](https://stackedit.io/).
