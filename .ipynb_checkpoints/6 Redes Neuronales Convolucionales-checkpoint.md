
# Redes Neuronales Convolucionales

Son un tipo de red neuronal para procesar y extraer características de datos que tienen forma de malla.

Son muy útiles procesando datos correlacionados. Por ejemplo: 

* Imágenes (correlación espacial),  
* Series de tiempo (Correlación temporal), 
* Secuencia de imágenes en el tiempo.  video, imágenes multi-temporales, series espacio temporales (Correlación espacio temporal).


Las redes convolucionales  utilizan una operación que se llama **Correlación cruzada**, la cual está relacionada con la  operación de **convolución**. 



### Convolución

Es una operación sobre dos funciones que reciben argumentos de valores reales.

[Ejemplo  mediciones de rastreo de ubicación en el tiempo]

Supongamos que tenemos un laser que mide $x$ en en cada instante de tiempo $t$ , $x(t)$, el cual genera una serie de tiempo. 

Estas mediciones son ruidosas y por lo tanto queremos reducir el ruido de la señal para estimar la posición real.

Aquí lo importante es que si queremos filtrar el ruido, debemos considerar que las mediciones mas recientes son mas importantes que las antiguas. Por lo que para estimar, requeriríamos de una función de peso $w(a)$ donde a es la "edad" de la medición.  Si aplicamos esta función de peso, obtendríamos una versión suavizada de la serie de tiempo $s(t)$.

Esta operación que necesitamos hacer se llama convolución y se define como:

$$s(t) = \int x(a)w(t-a) da$$

$$s(t)=(x \ast w)(t)$$

En el diseño del kernel para este caso se requieren dos consideraciones importantes:

* $w$ debe ser el resultado de una distribución de probabilidad válida (o sea en teoría desde $w(a - \infty)$,  a $w(a + \infty)$ sumen 1).
*  pero $w(a)$ ser negativo para argumentos menores que 0, o sea esos valores que representen valores en el futuro que no conocemos.

En general la convolución esta definida para cualquier función donde la integral esté definida.

En terminología convolucional $x(a)$ es la entrada, y $w$ es el kernel.

La versión discreta de la convolución es:

$$s(t) = (x \ast w)(t) =\sum_{a=-\infty}^{\infty} x(a)w(t-a)$$

En aprendizaje automático la entrada puede ser arreglos de cualquier dimensión llamados tensores.

Estas funciones de convolución pueden interpretarse como funciones con ceros casi en cualquier parte del dominio, excepto para un conjunto de valores.

Con esto podemos implementar la suma infinita como **suma sobre un conjunto finito de elementos**.

$$S(i,j) = (I \ast J)(i,j) = \sum_{m} \sum_{n} I(m,n) K(i-m,j-n)$$

Como la convolución es conmutativa, equivalente mente se puede escribir como:


$$S(i,j) = (K \ast  I)(i,j) =\sum_{m} \sum_{n} I(i-m,i-n) K(m,n)$$

Esta propiedad se presenta porque hemos *volteado (flipped)* el kernel con respecto a la entrada. 

Esta propiedad es la que motiva el uso de la correlación cruzada (cross-correlation). 

La operación estructuralmente es la misma pero sin *voltear* el kernel.

$$S(i,j) = (I \ast'  K)(i,j) =\sum_{m} \sum_{n} I(i+m,i+n) K(m,n)$$

Esto se puede escribir como:

$$ (I \ast'  K) =  (I \ast  \text{rotate180}^o(K))$$
[Ejemplo Libro fig 9.1]

Tipos de correlación cruzada:

1. Válido. El kernel nunca se sale de la entrada. La salida es mas pequeña que la entrada. 
2. Igual. (*same*). La salida como resultado de aplicar el kernel es igual que el tamaño de la entrada. ( Más detalls sobre como Tensor Flow produce esta salida https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python)
3. Completo (*full*). El kernel se "sale" de la entrada. O en otras palabras, el kernel se aplica tan pronto y genere una intersección en los datos de entrada. La salida es mas grande que la entrada.

**Propagación hacia adelante de la red Neuronal convolucional**:

Para un kernel expresa como 

$$S = X \ast ' K +B$$

Para mas de un kernel expresa como 

$$S_d = X \ast ' K_{d} +B_d$$



Para más de un kernel y una entrada con mas de una dimensión expresa como. 


$$S_d = \sum_{j=1}^n X_j \ast ' K_{d,j} +B_d$$
Pregunta: ¿Cómo sería la propagación hacia atrás de esta red convolucional?

**Ejemplo con series de tiempo**

$$[x_1,x_2,x_3,x_4,x_5] \ast' [k_1,k_2] = [x_1k_1+x_2k_2,x_2k_1+x_3k_2,\dots,x_4k_1+x_5k_2]$$

**Ejemplo con imágenes**


![enter image description here](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/convolution-with-multiple-filters2.png?raw=true)


![enter image description here](https://i.ibb.co/sKryDyZ/example-cc-operation.png)



**Motivación**

La convolución aprovecha tres ideas importantes:

1. Interacciones dispersas. 

Se genera con un kernel más pequeño que la entrada. 

Su función es detectar características importantes de manera compacta.

Implica guardar pocos parámetros. El cálculo se vuelve menos complejo para producir una salida. 

En la Fig 9.2 se denota que no todas las salidas $s$ son afectadas por una unidad de entrada $x$, a diferencia de las redes neuronales de propagación hacia adelante dónde la entrada $x$ esta conectada con todas las salidas $s$.

En la Fig 9.3 se observa las entradas $x$ que afectan a la salida $s$. Estas dependen del tamaño del kernel. Se les llama *capto receptivo*

En Fig 9.4. En redes neuronales convolucionales profundas, el campo receptivo es mas grande que en las redes convolucionales *superficiales* o poco profundas.

Esta estructura permite que las salidas interactuen con la mayoría de la entrada de datos. (ya sea una imagen, series de tiempo, etc...)


(Fig 9.2-9.4)



2. Compartición de parámetros.

Se refiere al rehuso de los parámetros para mas de una función en el modelo. Un kernel $K \in R^{m,n}$  utilizado varias veces. El kernel se desplaza con una ventana deslizante sobre los datos para hacer la operación. A esto de le llama **pesos atados** (tied weights).



3. Representaciones equivariantes

Si la entrada cambia, la salida cambia de la misma manera.

Es importante que la convolución no siempre es equivariante a algunas otras transformaciones como escalamiento y rotación.

### *Pooling*


**Etapas de la red Convolucional**

1. Convolución en paralelo que producen un conjunto de activaciones lineales. (Le hemos llamado pre activación)
2. Etapa de detección: Consiste en la activación del resultado de la etapa uno. 
3. Aplicación de una función que proporciona una medida estadística representativa de una conjunto de datos de salida. de *pooling* como pooling máximo, 
pooling promedio
pooling $L^2$
pooling  usando un promedio ponderado basado en la distancia con respecto al pixel central.

Pooling extrae información que es poco sensible a las traslaciones de los datos. 

Como pooling resume la respuesta sobre una vecindad, es posible usar menos unidades de pooling que unidades de detección al deslizar el kernel cada $k$ pasos en vez de uno. Esto genera un submuestreo con menos datos que aquellos presentados en la capa de detección. (Fig 9.10)



**Convolución en paralelo**




Referencias
https://www.youtube.com/watch?v=Lakz2MoHy6o
https://www.juanbarrios.com/redes-neurales-convolucionales/
https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0
https://www.tensorflow.org/tutorials/images/cnn

> Written with [StackEdit](https://stackedit.io/).
