
# Aprendizaje Automático parte II

# Redes neuronales

* Una red neuronal Artificial es un modelo matemático inspirado en el comportamiento biológico de las neuronas y en la estructura del cerebro. 
* Las ideas y descubrimientos en neurociencias sirven como inspiración en la construcción de modelos matemáticos - computacionales capaces de resolver problemas complejos de clasificación y regresión como son las redes neuronales artificiales. 
* Un esquema simplificado de una neurona biológica de un tipo se muestra en la siguiente figura.

![Screen Shot 2021-08-19 at 18.30.46](https://i.ibb.co/JjMrzGR/dendrita.png)

* La figura muestra las partes de una neurona que son de interés para la construcción de estos modelos matemáticos.
  * SOMA: Cuerpo central de la neurona la cual procesa la información de entrada y la transfiere al axón.
  * Axón: es la prolongación del SOMA . 
  * Dendritas: reciben información que transfieren al SOMA.
  * Sinapsis. Zona de conexión entre una neurona y otra.

La neurona transmite los impulsos nerviosos a otras neuronas interconectadas. Estos viajan desde las dendritas hasta el axón. El axón se comunica con otras neuronas por sinapsis (comunicacación asoma-dendrita) construyendo de esta manera una red de comunicación neuronal.

 La sinapsis constituye el sitio físico que sirve de puente para el paso de información de una neurona a otra, permitiendo que las diferentes partes del sistema interactúen funcionalmente.

Este modelo simplificado ha servido de inspiración para la construcción de estos modelos.

El primer modelo neuronal llamado modelo McCulloch-Pitts surgió en 1943 fué desarrollado por: 

* El psiquiatra y neuroanatomista Warren McCulloch y 
* El Matemático Walter Pitts.

Se apoyó de las matemáticas para simular y explicar el comportamiento de la neurona. 

Este modelo neuronal es bastante simple. 

Es una combinación lineal de un vector entrada de dos dimensiónes al cual se le suma una constante (bias) , y regresa una salida.



El experimento demuestra una analogía donde:

* El vector de entrada $\mathbf{x}^T = [x_1,x_2]$ representa un estímulo de un entorno externo que recibe la neurona artificial.

* La salida $z$ es la respuesta del estímulo. 

* La salida $z$ se va adaptando a su entorno al ajustar los pesos sinápticos $W^T = [w_1,w_2]$ y el termino aditivo o "bias".

En matemáticas son los parámetros optimizables de un modelo de regresión.

Matemáticamente el modelo se describe como 

$$z =  f (w_1x+w_2y+b) = f (\mathbf{W}^T \mathbf{x}+b)$$ (X es vector columna y W es un vector fila)



$f$ es la función de activación que simula el procesamiento que se produce en el soma. 

Así nace el *perceptrón*! que es técnicamente lo mismo que la regresión logística (lineal) con la diferencia que este no se limita a solo dos entradas impuesta por el modelo de McCulloch-Pitts 



### Perceptrón Multicapa



La base del perceptrón multicapa son los perceptrones simples.

Estos perceptrones simples estan interconectados entre si dando lugar a diferentes tipos de arquitecturas en redes neuronales como:



![neuralnetworks](https://i.ibb.co/N6Rrg2r/neuralnetworks.png)

Un perceptrón Multicapa con propagación hacia adelante tiene la característica de  tener sus neuronas organizadas por capas y cada capa se limita a transferir la información hacia adelante desde la capa de entrada hasta la capa de salida.



![DFFN](https://i.ibb.co/ZLCGc1Z/DFFN.png)

* Una red neuronal de propagación hacia adelante es una función universal de aproximación con 3 tipos de capas:
  * Capa de entrada: es una sola y recibe un vector de entrada de algún tamaño.
  * Capas intermedias: puede contener a partir de una, y cada capa puede conteneer varias funciones de activación llamadas neuronas. 
  * Capa de Salida: Es una sola capa y puede tener multiples neuronas de salida.



El entrenamiento de este tipo de funciones al igual que los modelos lineales de regresión y clasificación esta compuesto por:



1. Evaluación
2. Corrección

* Evaluación:

  * Evaluación del modelo dado un conjunto de parámetros $\theta$ utilizando una función de costo $C(\theta)$

* Corrección:

  * Calculo de la Derivada del modelo.

    * Ejecutar el algoritmo adecuado. En redes neuronales se utiliza un algoritmo llamado backpropagation.

    

    $\hat{y}= f(g(x,\theta_g),\theta_x)$

    $\frac{{\partial \hat{y} }}{{\partial {w \in {\theta_g \theta_x} }}}$

    * Actualización de los parametros $\theta$. En redes neuronales esto se hace con un algoritmo que propaga el error de la última neurona hacia atras.



Evaluar una función que representa una red neuronal tiene sus consideraciones.





## Regla de la cadena del cálculo

La regla de la cadena del cálculo es utilizada para calcular derivadas de funciones compuestas (e.g., $y = f(g(x))$) por otras funciones de las cuales se conocen sus derivadas.

La regla de la cadena es fácil de escribirla algebraicamente. 



Ejemplo:

Sea $x$ un número real, y $f$ y $g$ funciones de mapeo. Supongamos que tenemos las funciones anidadas ($g o f$) $z=f(g(x))$ y queremos calcular su derivada parcial por la regla de la cadena.

 

$$z = f(g(x))$$

$y=g(x)$

$$z = f(y)$$

 La regla de la cadena nos dice que para calcular la derivda de $z$ podemos hacerlo de esta manera:

 $\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}$

Esto se puede generalizar para calcular funciones que reciben vectores. 

Supongamos que $\mathbf{x}\in {R}^m$ , $\mathbf{y}\in R^n$, $g$ mapea de $R^m$ a $R^n$, y $f$ mapea de $R^n$ a $R$. Si $\mathbf{y}= g(\mathbf{x})$ y $z = f(\mathbf{y})$

$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$



El algoritmo propagación hacia atrás utiliza el principio de la regla de la cadena. De tal manera que el cálculo de la derivadada parcial de redes neuronales con muchas neuronas (y por lo tanto parámetros), sea eficiente guardando parte del cálculo de la derivada númerica en los nodos.



**Ejemplo Formulación de Propagación hacia atrás (backpropagation) con una neurona para resolver un problema de regresión lineal con la regla de la cadena**

Modelo de regresión lineal sin sesgo.

$$\mathbf{w}^T \mathbf{x}_i  = \hat{z}_i$$

donde el error está dado por $e_i = \frac{1}{2}(z_i-\hat{z}_i)^2$


Las operaciones que se realizan (de atrás hacia adelante) son

Cálculo del error en el conjunto de entrenamiento. 

$E= \frac{1}{2} \sum_{i\in \text{training}}^n (z_i-\hat{z}_i)^2$

Derivada del Error 

$\frac{\partial E}{\partial \hat{z_i}} = - \sum_{i \in \text{training}}^n (z_i-\hat{z}_i)$

Cálculo de la salida $\hat{z}_i$. 

$\hat{z}_i = \mathbf{w}^T \mathbf{x}_i$

Cálculo de la derivada parcial de la salida  $\hat{z}_i$ con respecto a $w_j$

$\frac{\partial z_i}{\partial {w}_j} =  {x}_{j,i}$



$$\nabla_\mathbf{w} \hat{z}^T = [\frac{\partial \hat{z}_i}{\partial {w}_1},\frac{\partial \hat{z}_i}{\partial {w}_2}] $$

Para calcular la derivada del error con respecto a los parámetros $\mathbf{w}^T=[w_1,w_2]$, 

$\frac{\partial E}{\partial {w_j}} = \frac{\partial E}{\partial \hat{z_i}} \frac{\partial z_i}{\partial  w_j}$ 


$\frac{\partial E}{\partial {w_j}} =  - \sum_{i}^n (z_i-\hat{z}_i){x}_{j,i}$ 

$$\nabla_\mathbf{w} E^	T = [\frac{\partial E}{\partial {w}_1},\frac{\partial E}{\partial {w}_2}] $$



**Diseño de la Arquitectura de la red neuronal multicapa, el caso de la red neuronal profunda multicapa.** 

La primera capa puede ser expresada

$$h^{(1)} = g^{(1)}(W^{(1)T}X+b^{(1)})$$

Las capas intermedias tienen la siguiente estructura:

$$h^{(2)} = g^{(2)}(W^{(2)T}h^{(1)}+b^{(2)})$$

Esta expresión puede representarse graficamente.



Hasta ahora hemos practicado con arquitecturas muy simples. 

Para calcular el gradiente de este tipo de red neuronal hacia adelante con varias capas intermedias, se requiere considerar un modelo dependiente de los parámetros (la respuesta del modelo depende de los valores de los parámetros). 


**Modelo dependiente de los parámetros **

![enter image description here](https://i.ibb.co/TMv2p3Y/parameter-Dependet-Model.png)

Ejemplo:

Calcular el gradiente $\nabla_W z$ del siguiente modelo.



![enter image description here](https://i.ibb.co/YXcGXD1/Example-Parameter-Dependet-Model.png)

Para $\mathbf{Y}$, las derivadas parciales son:
 
$$(\nabla_{\mathbf{Y}}{z})^T =\left[\frac{\partial {z}}{\partial y_1},\frac{\partial z}{\partial y_2} \right]$$

Para cada elemento $w_i$ en $\mathbf{W}$ la derivada parcial es equivalente a:

$$\frac{\partial{{z}}}{ \partial w_i} = \sum_{j=1}^2 \frac{\partial {z}}{\partial y_j} \frac{\partial y_j}{\partial w_i}$$


Esto se puede escribir de una manera mas compacta, como el producto de la matriz Jacobiana transpuesta $\frac{\partial \mathbf{y}}{\partial {W}}^T$ por el gradiente $\nabla_{\mathbf{y}}{z}$:

$$\nabla_{W} {z} = \left( \frac{\partial \mathbf{y}}{\partial {W}}\right)^T \nabla_{\mathbf{y}}{z}$$

Esta es la operación detrás del algoritmo de propagación hacia atrás.

**Algoritmo de propagación hacia adelante con múltiples capas intermedias (*Multi Layer Perceptron*)**

![enter image description here](https://i.ibb.co/M26CgjL/forward-prop.png)

**Algoritmo de propagación hacia atrás**.

![enter image description here](https://i.ibb.co/1bWVvMZ/backprop-alg.png)
1) Teorema de la aproximación Universal. (Impulsó la motivación del estudio de redes neuronales)

"Las redes neuronales con propagación hacia adelante con una salida lineal (Hornik et al,. Cybenko et al., 1989) y al menos una capa oculta con alguna cualquier función no lineal aplastada (*squashing function*) e.g., una de tipo sigmoidal), puede aproximar cualquier función (Medible por Borel)  continua y acotada en cualquier dimensión $\R^n$ con **cero error de aproximación**, siempre y cuando la red neuronal tenga suficientes neuronas.

En 1990, Hornik et al., probó que las derivadas de este tipo de redes neuronales también se aproximan a las derivadas reales.

En 1993 (Leshno et. al. 1993) se probó que también este teorema se sostiene usando otro tipo de funciónes como ReLu.

Aunque es un teorema muy fuerte que motiva el estudio de redes neuronales, no dice que tan grande tiene que ser el modelo para aproximar (Barron 1993).

Si una red neuronal con propagación hacia adelante es un aproximador universal, ¿Porqué estudiar modelos profundos?

R: Porque la profundidad de los modelos permite reducir el número de parámetros de optimización sin sacrificar capacidad de representatividad.

El teorema de Montufar en 2014, apoyó  la motivación del estudio de arquitecturas profundas.  Dice que el numero de regiones lineales que pueden sacarse de un rectificador lineal, con 

* $d$ entradas
* $l$ capas de profundidad
* $n$ unidades está dado por:

$$O \left( \left( \overset{d}{n} \right)^{d(l-1)} n^d\right)$$

Consideraciones:

* Escoger el número de capas ocultas.
* Escoger el tamaño de cada capa
* La arquitectura ideal para una tarea de clasificación debe ser buscada revisando y monitoreando su error de generalización con validación cruzada. 

Propiedades de Aproximación Universal y profundidad.



El teorema de la aproximación universal de Hornik 1989 prueba que una red de propagación hacia adelante con una entrada lineal y con al menos una capa oculta compuesta por alguna función no lineal puede aproximar cualquier función con dimensión finita a otro con cero error, siempre y cuando la red neuronal tenga suficientes neuronas.

Si una sola capa oculta puede aproximar cualquier función, Entonces porque estudiar redes neuronales con propagación hacia adelante con mas de una capa intermedia? 

























> Written with [StackEdit](https://stackedit.io/).