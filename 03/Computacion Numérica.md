# 3. Computación numérica



Los algoritmos de Aprendizaje automático resuelven problemas matemáticos a través del tratamiento numérico intensivo.

Esto es que de manera no analítica tratan de resolver problemas de manera iterativa. 
Se requieren repetir muchas veces una operación para encontrar o aproximar una solución.


Operaciones comunes que requieren uso intensivo de la computación numérica estan relacionadas con:
	* Algebra lineal (resolución de sistemas de ecuaciones lineales).
	* **Optimización** (Encontrar los mejores parámetros de un modelo.) 

La primer dificultad fundamental que encontramos, es la representación limitada que las computadoras tienen con números enteros y reales.

Con los números enteros es problemática la representación de números muy grandes y con los números reales la precisión limitada debido al diseño que tienen las computadoras digitales: Tenemos un número finitio de bits para representar un conjunto infinito de números reales, por lo que es común que la computadora recurra al truncamiento o redondeo de las cifras. 





**Desbordamiento**


*  Redondeo por cero, o computación de división entre numeros muy pequeños.
*  La representación de números muy grandes con $\pm \infty$, en algunos lenguajes de programación es tratado como un no número y genera problemas graves de calculo,  cómo la función softmax usada comunmente para predecir probabilidades de pertenencia a alguna clase en problemas de clasificación multi-clase.

	


$$\text{softmax}(\mathbf{x})_i =  \frac{\text{exp}(x_i)}{\sum_{j\in{\{1,\dots
,n\}}} \text{exp}(x_i)}$$


El cálculo incorrecto se puede dar cuando al menos un elemento de $\mathbf{x}$ es muy grande, generando un $\infty$ que es tratado como un $NaN$ y por lo tanto genera una suma incorrecta en el denominador.

La solución es cambiar la variable de tal manera que softmax no cambie su valor de salida. Esto es que:

$$\text{softmax}({\mathbf{x}})_i = \text{softmax}({\mathbf{z}})_i $$

se puede lograr con $\mathbf{x}$ con $\mathbf{z}=\mathbf{x}-\text{max}_i x_i$. 



En general las librerías especializadas se encargan de manera transparente de estos problemas. Sin embargo es necesario tenerlos en cuenta para futuras implementaciones de nuevas propuestas con operaciones que tengan estos problemas.

	
**Condicionamiento**

Es una propiedad de ciertas operaciones como las matriciales que se ven afectadas por el error de medición, redondeo o truncamiento.

*  	En la resolución de sistemas de ecuaciones lineales, el número de condición es una medida para calificar la estabilidad númerica de algunas operaciones. Algunas operaciones son mas sensibles a los errores de truncamiento que otros.
*   Por ejemplo, si tenemos que resolver $Ax = b$, y el número condicion de $A$ es muy grande, entonces la solución verdadera $x$ estará alejada de la realidad.  Esto se acentúa si $b$ proviene de mediciones poco precisas.
*   
Es un número que nos indica la sensibilidad a las operaciones numéricas de una función.

En álgebra lineal, con la función $b = \mathbf{A}^{-1}{x}$ con $A\in {n \times m}$ tiene una descomposicion de eigenvalores, su número condición esta dado por la razón de su eigenvalor mas grande con su eigenvalor mas pequeño.

$$\text{max}_{i,j} = |\lambda_i/\lambda_j|$$

E.g., si este número es grande, entonces la salida de  $\mathbf{A}^{-1}{x}$ es muy sensible al error de truncamiento en $x$.

Esto es una propiedad matricial con la cual debe de lidiar el cómputo numérico. 



**Optimización basada en gradiente**

Optimización es la tarea de encontrar el valor máximo o mínimo de una función $f(\mathbf{x})$ probando diferetes valores de $\mathbf{x}$.

Optimización está involucrada en TODOS los modelos de aprendizaje automático y profundo. (¿Alguien conoce algún método que no?). 


Usualmente los problemas se definen con respecto a la minimización, y para maximizar se utiliza esta equivalencia.

$$\underset{x}{\text{max }} f(\mathbf{x}) = \underset{\mathbf{x}}{\text{min }} - f(\mathbf{x})$$


La función  $f$ que queremos optimizar su valor se le llama **función objetivo, criterio, función de costo, función de perdida, función de error**.

A veces nos interesa mostrar el argumento que minimiza la función.

$$\mathbf{x}^* = \underset{\mathbf{x}}{\text{arg min }} f(\mathbf{x})$$


Hay dos tipos de optimización **Local y Global**

En optimización **Local** queremos encontrar $ \{x^*: f(x^*) < f(x), \forall x, x \neq x^* \wedge ||x-x^*|| < \epsilon \}$.

En optimización **Global** queremos encontrar $\{x^*: f(x^*) < f(x), \forall x, x \neq x^*$. \}



Los problemas se pueden clasificar en **Convexos** y **No Convexos**). Estos se aplican a funciones y conjuntos.




**Definición de un problema convexo.**

Formalmente, para dos puntos  cualquiera $x_0,x_1$ en el conjunto $S$ , tenemos el conjunto de puntos definido por:

$$\alpha x_0 + (1-\alpha)x_1 \in S \text{ para }  \alpha \in [0,1]$$.

Donde podemos decir si la función $f$ es convexa si cumple con la siguiente desigualdad

$$f(\alpha x_0 + (1-\alpha)x_1) \leq \alpha f (x_0) + (1-\alpha)f(x_1), \\\text{ Para toda } \alpha \in [0,1]$$

En los problemas convexos se tiene garantía que el óptimo local es también un óptimo global.

No es así en los problemas no convexos como aquellos que se presentan en aprendizaje profundo. 

El objetivo de los problemas de optimización es *mover* de la mejor manera $\mathbf{x}$ para encontrar el mínimo dado un problema de optimización.

La expresión general de actualización de $x$ para una función unidimensional
$$x_{t+1} = x_t + \epsilon  \cdot f'(x) $$

La cual podemos observar 3 componentes (correcciones):

* El vector $x_t$, (que puede verse como un vector de estado en el tiempo $t$).
* $f'(x)$: El valor de la derivada. Indica **la dirección** a mover para $x$, y sugiere una magnitud de desplazamiento.
* $\epsilon$: Es la tasa de aprendizaje. Indica cuanto "le haremos caso" a la magnitud de la derivada.  Es un número de preferencia pequeño menor que 1 $f'(x))$.


Idealmente queremos que siempre por cada iteración mejorar la solución previa. $f(x_{t+1})< f(x)$.  Pero no es posible garantizarlo.

Un tipo de optimización muy utilizada en aprendizaje profundo es optimización basada en gradiente.

Esto es porque la funciones objetivo por lo regular son continuas y derivables y se explota esta característica para saber hacia donde mover los parámetros.


**Gradiente**

¿Porqué es buena idea considerar el gradiente en problemas de optimización?


1. Nos da información sobre la distanca que nos separa del óptimo.
2. Nos da información hacia donde debemos mover la solución actual $x_t$.

El algoritmo de gradiente descendente nos dice que $x_t$ debe moverse en dirección contraria al gradiente. Esto es:, 


$$x_{t+1} = x_t - f'(x)$$

**Puntos críticos o estacionarios**

Son aquellos **mínimos, máximos, puntos de silla** donde la derivada $f'(x) = 0$. (Goodfellow, pág 81, Fig 4.2,)


En aprendizaje automático buscaremos una solución, la cual nos permita al menos obtener un muy buen óptimo local.


Para entender como funciona el algoritmo basado en gradiente, optimizaremos funciones de múltiples entradas, una sola salida.

$$f: \mathbf{R}^n \rightarrow \mathbf{R} $$.


**Generalización de la derivada en dimensiones mayores a 1**

Cuando $\mathbf{x} \in \mathbf{R}^{n}$ necesitamos calcular las derivadas parciales $\frac{\partial}{\partial x_i} f(\mathbf{x})$de cada entrada $x_i \in \mathbf{x}$. con el objetivo de medir cuando cambia $f$ en funcion de una perturbación en $x_i$. El vector de estas derivadas parciales está dado por:


$$[\frac{\partial f(\mathbf{x})}{\partial x_0}, \frac{\partial f(\mathbf{x})}{\partial x_1},\dots \frac{\partial f(\mathbf{x})}{\partial x_{n-1}}]^T =\nabla_{\mathbf{x}} f(\mathbf{x}) $$


### Algoritmo de Descenso por Gradiente

Derivación del algoritmo.


Para una sola dimensión es sencillo intuir que la dirección contraria a la derivada es donde se encuentra el descenso. Para generalizar la idea en una función de mas de una dimensión, la dirección que nos da el descenso mas pronunciado se prueba de la siguiente forma.

Dada una función evaluada en $\mathbf{x}$, como $f(\mathbf{x})  \in R^n $, ¿A qué dirección se encuentra la pendiente mas pronunciada? 

Para contestar esta pregunta, una manera de responder es formulando nuestro problema de una manera adecuada.

$$\underset{\mathbf{u^Tu}=1}{\text{min }} \frac{\partial }{\partial \alpha} f(\mathbf{x} + \alpha \mathbf{u})$$

donde:
 
* $\alpha$ es un valor muy pequeño que tiende a 0.
* $\mathbf{u}$ es el vector unitario que nos indica la dirección a la cual se movería $\mathbf{x}$ 


Primero Queremos resolver la expresión que nos da la pendiente para un desplazamiento muy pequeño $\alpha \rightarrow 0$ de $\mathbf{x}$ a una dirección $\mathbf{u}$. A esto se le llama la **Derivada Direccional**.


$$ \frac{\partial }{\partial \alpha} f(\mathbf{x} + \alpha \mathbf{u})$$

Con la regla de la cadena hacemos cambio de variable:

$$y = f(\mathbf{x}+\alpha \mathbf{u})$$


$$v = \mathbf{x}+\alpha \mathbf{u}$$

$$y = f(\mathbf{v})$$

Resolvemos:

$$\frac{\partial y}{\partial \alpha} = \frac{\partial y}{\partial v}  \cdot \frac{\partial v}{\partial \alpha}$$

$$= \nabla_{\mathbf{v}} f(\mathbf{v}) \cdot \mathbf{u}$$

Equivalentemente sustituimos el producto punto y reacomodamos términos en forma de multiplicación matricial.

$$= \mathbf{u}^T  \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x}+\alpha \mathbf{u}})  $$

Evaluamos la derivada para $\alpha = 0$

$$= \mathbf{u}^T  \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x}})  $$

Una vez que resolvimos la derivada, el siguiente paso es encontrar la dirección (dado por la dirección del vector unitario) que haga la función $f$ decremente lo más rápido. En otras palabras que nos minimice la expresión del gradiente.  

$$\underset{\mathbf{u},\mathbf{u}^T \mathbf{u}  1}{\text{min }} = \mathbf{u}^T \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x}})  $$

Esto es esquivalente a encontrar el ángulo que debe existir entre $\mathbf{u}$ y $ \nabla_{\mathbf{x} } f (\mathbf{x})$ que minimice la expresión anterior.

Recordemos que el producto punto entre dos vectores, es igual a multiplicar su magnitud y el coseno del angulo entre ellos.

$$\underset{\mathbf{u},\mathbf{u}^T \mathbf{u} = 1}{\text{min }}   || \mathbf{u}||_2 || \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x})}||_2 \text{ cos } \theta $$


Recordemos que: 
$$||\mathbf{a}||_2 ||\mathbf{b}||_2 \text{ cos } \theta=  \frac{\mathbf{a} \cdot \mathbf{b}}{ ||\mathbf{a}||_2 ||\mathbf{b}||_2}$$


Como $\mathbf{u}^T \mathbf{u} =1$, y el gradiente no influye en el proceso de minimización por ser considerada una expresión constante. 



$$\underset{\mathbf{u},\mathbf{u}^T \mathbf{u} = 1	 }{\text{min }}   \frac{\mathbf{u} \cdot \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x})}}{||\mathbf{u}||_2 ||\nabla_{\mathbf{x}} f (\mathbf{x})||_2 }$$

El vector $\mathbf{u}$ que minimiza es aquel que genera un ángulo de 180 grados ($ \pi$) con respecto al vector de las derivadas parciales.




Por lo tanto la dirección donde se encuentra la pendiente en descenso mas pronunciada **es justo en la dirección contraria del a la pendiente dada a la derivada  parcial**. A este método se le llama **Descenso de Gradiente o Descenso mas pronunciado**.


$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \epsilon \nabla_\mathbf{x} f (\mathbf{x}) $$

Donde $\epsilon$  es la tasa de aprendizaje, un escalar positivo que define el tamaño de paso. El algoritmo converge cuando $\nabla_\mathbf{x} f (\mathbf{x})$ es cero o muy cercano a cero. Algunas ocasiones podemos evitar las iteraciones y encontrar directamente la solución $\mathbf{x}$ que $\nabla_\mathbf{x} f (\mathbf{x})=0$.


Alguna estrategia para escoger $\epsilon$ pueden ser: 

1. Escoger un valor muy pequeño.
2. Hacer algún tipo de búsqueda lineal hasta encontrar un valor de $\epsilon$ que minimize la función objetivo.


$$\underset{\epsilon, \epsilon \leq 1}{\text{ min }} f(\mathbf{x}_t - \epsilon \nabla_\mathbf{x} f (\mathbf{x}) )$$

### **Derivada, 2da Derivada y su generalización en altas dimensiones**

**Problemas con el algoritmo de gradiente descendente**

Ver Fig. 4.4 (Goodfellow, pag 84)
1. El gradiente solo predice de una manera precisa funciones que se comportan linealmente (en su totalidad o en algún intervalo, donde no tienen curvatura). (e.g. y = mx +b).
2. Con funciones no lineales pueden suceder dos cosas:  
	1 . Si la curvatura es negativa, el valor de $f(\mathbf{x})$ se decrementará mas rápido que el estimado por el gradiente  $f(\mathbf{x_t}-\epsilon  \nabla_{\mathbf{x}}f(\mathbf{x}))$.
	2 . Si la curvatura es positiva  $f(\mathbf{x})$ el valor decrementara mas lento que lo estimado por gradiente.
	

**Solución:**

Utilizar el concepto de segunda derivada para descubrir si existe curvatura.

Ejercicios, verificando curvaturas con 2da derivada de funciones univariadas:

1. Dado $f(x) = mx+b$, ¿Cuál es su derivada para dos valores diferentes de $x$? 
2. Con respecto a la función anterior, cual es su segunda derivada? 
3. Dado $f(x) = (mx+b)^2$, encontrar de manera analítica:
	1. La derivada $f'$ y la solución $x$ que resuelve $ f'(x)=0$ 
	2. Calcular la 2da derivada $f''(mx+b)^2$ y evaluar en $f''(x)$ donde $f'(x)=0$.

3. Lo mismo que el anterior pero con probando para $f(x) = -(mx+b)^2$.

	

**Jacobiano**

Para calcular la derivada parcial de una función que recibe vectores de tamaño $m$ y devuelve  vectores $n>1$. $f: \mathbb{R}^m \rightarrow \mathbb{R}^n$, En cálculo de vectores, es necesario calcular las derivadas parciales: $$J_{i,j}=\frac{\partial}{\partial x_j} f (\mathbf{x})_i$$ 



![](https://wikimedia.org/api/rest_v1/media/math/render/svg/cd4cfdd4fd3a250f3bb15bc6b06372ea3a2da65f)

**Hessiano**

Es la versión vectorial de la segunda derivada.

Se expresa como: $$H_f(\mathbf{x})_{i,j} = \frac{\partial^2}{\partial x_i \partial x_j}f(\mathbf{x})$$

Y en su forma matricial es:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/16edfd41354f9adbe656c26498348a8bee5d6c7b)


Como el Hessiano es una matriz  simétrica, podemos descomponerla en un conjunto de eigenvalores reales y bases ortogonales de eigenvectores.

La segunda derivada nos da información sobre el desempeño del algoritmo de Descenso de Gradiente observando al curvatura aproximada y sobre eso ajustar un valor a $\epsilon$.

Para descubrir ese valor se recurren a heurísticas.

**Lo que veremos en la clase de hoy**
1. Serie de Taylor y búsqueda de $\epsilon$
2. Como detectar si $f'(x)=0$ es un máximo, mínimo o punto de slla
3. Método de Newton para optimización.
4. Función lipschitziana.

**Un algoritmo usando series de Taylor para buscar el tamaño de paso valor $\epsilon$**



La serie de Taylor esta definido como una suma de términos de la forma:

$$f(x)=\sum^\infty_{n=0} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

Es una herramienta poderos para aproximar una función que no conocemos $f(\mathbf{x})$ alrededor de un punto $x_0$. 



$$f(\mathbf{x}) \approx f(\mathbf{x_0})+(\mathbf{x}-\mathbf{x}_0)^T \mathbf{g} + 1/2 (\mathbf{x}-\mathbf{x}_0)^T \mathbf{H} (\mathbf{x}-\mathbf{x}_0)$$

donde:
 
$\mathbf{g}$ : Es el gradiente evaluado en $\mathbf{x}_0$.   
$\mathbf{H}$: El Hessiano evaluado en $\mathbf{x}_0$. 

******
**Ejercicio, Generar la serie de Taylor para $y = x^2$**
******

Podemos utilizar la serie de Taylor para calcular un buen valor para la tasa de aprendizaje $\epsilon$. Para esto 
sustituimos $\mathbf{x}$  por $\mathbf{x_0}-\epsilon \mathbf{g} $. Tenemos:

$$f(\mathbf{x}_0 - \epsilon \mathbf{g}) \approx f(\mathbf{x}_0)+((\mathbf{x_0}-\epsilon \mathbf{g} )-\mathbf{x}_0)^T \mathbf{g} + 1/2 ((\mathbf{x_0}-\epsilon \mathbf{g} )-\mathbf{x}_0)^T \mathbf{H} ((\mathbf{x_0}-\epsilon \mathbf{g} )-\mathbf{x}_0)$$

Simplificando:

$$f(\mathbf{x}_0 - \epsilon \mathbf{g}) \approx f(\mathbf{x}_0)-\epsilon \mathbf{g}^T \mathbf{g} +1/2 \epsilon^2 \mathbf{g}^T H \mathbf{g}$$

Vemos 3 componentes.

1. El valor original $f(x_0)$. 
2. La mejora esperada dada la pendiente de la función
3. La corrección tomando en cuenta la curvatura de la función.

Cuando $\mathbf{g}^T \mathbf{H} \mathbf{g}$ es positiva podemos resolver el tamaño de paso que mejor decrementa la aproximación de $f$ con series de Taylor. 

$$\underset{\epsilon}{\text{ arg min }} j(\epsilon)$$

donde  $j(\epsilon) = f(\mathbf{x}_0)-\epsilon \mathbf{g}^T \mathbf{g} +1/2 \epsilon^2 \mathbf{g}^T H \mathbf{g}$

Resolver numéricamente es muy costo. e.g. con Descenso de gradiente??!. Podemos  encontrar la solución exacta a esta expresión.

1. Resolver la derivada de la expresión 
2. Encontrar aquella $\epsilon$ que haga 0 la expresión

Solución
$$\epsilon^* = \frac{\mathbf{g}^T\mathbf{g}}{\mathbf{g}^TH\mathbf{g}}$$



**Significado de la Eigendescomposición de la matriz Hessiana**


| Segunda derivada en puntos críticos $f'(x)=0$ | Ejemplo     |    Hessiano en puntos críticos $ \nabla_x  f(\mathbf{x}) = 0$    |          Ejemplo            |                          Diagnóstico local de la función                          |  |
|:---------------------------------------------:|-------------|:----------------------------------------------------------------:|----------------------|:---------------------------------------------------------------------------------:|---|
|                   $f''(x)>0$                  | $f(x)=x^2$  | Todos eigenvalores $\lambda_i$  positivos                        | $f(x,y)= x^2+y^2$    | $x$ esta en un mínimo                                                             |   |
|                   $f''(x)<0$                  | $f(x)=-x^2$ | Todos eigenvalores $\lambda_i$  negativos                        | $f(x,y)= -(x^2+y^2)$ | $x$ está en un máximo                                                             |   |
|                   $f''(x)=0$                  | $f(x)=mx+b,f(x)=x^3$ | Al menos un eigenvalor es cero y los demás tienen el mismo signo |             ??  | ??
        | |   |
|                                               |             | Unos eigenvalores son positivos y otros negativos                | $f(x,y)= (x^2-y^2)$  | ?                                                                                 |   |

**Método de Newton**

Descenso de gradiente tiene un problema, cuando el número de condicionamiento de la matriz Hessiana es grande, es dificil encontrar un buen tamaño de tasa de aprendizaje $\epsilon$ para hacer mejoras significativas. Para esto podemos resolver la serie de Taylor en el punto crítico considerando hasta el 3er termino. De nuevo tenemos las serie de Taylor $T(x) \approx f(x)$

$$T(\mathbf{x}) = f(\mathbf{x_0})+(\mathbf{x}-\mathbf{x}_0)^T \mathbf{g} + 1/2 (\mathbf{x}-\mathbf{x}_0)^T \mathbf{H} (\mathbf{x}-\mathbf{x}_0)$$

Resolviendo para el punto crítico $T'(\mathbf{x}^*)=0$ (T de Taylor). y después despejando $\mathbf{x}^*$


Si $f$ es una función cuadrática positiva definida (matriz simétrica con valores reales), con el método de Newton brincamos a la solución directamente, si $f$ es no es una función cuadrática pero puede ser aproximada como una función cuadrática positiva definida, entonces  varias iteraciones serían requeridas para aproximarse al óptimo local.

Esto es mas eficiente que descenso de gradiente cuando estamos cerca de un mínimo local para la familia de funciones mencionada. 

**Sin embargo, no es muy bueno cuando estamos en un punto de silla, porque es atraido a estos**. En este caso descenso por gradiente no es atraido por estos puntos, pero puede apuntar por accidente y quedarse atascado ahí.


Método de newton se clasifica como un algoritmo de optimizacion de segundo orden. Por la 2da derivada que se le llama también de 2do orden o orden 2.

Por lo tanto siguiendo esta terminología, el método de Descenso de gradiente es un algoritmo de optimización de primer orden.


**El continuo de  Lipschitz**

Los algoritmos para aprendizaje automatico y profundo, se aplican para una familia muy amplía de problemas pero con muy poca garantía de convergencia.

Sin embargo existen algunas garantías restirngiendo problemas o funciones que son **Lipschitz continuas** o tienen derivadas continuas de Lpipshit

Una función continua de Lipschitz es aquella la cual su tasa de cambio es acotada por la **constante de Lipschitz** $L$

$$\forall \mathbf{x},\forall \mathbf{y}, |f(\mathbf{x})-f(\mathbf{x})| \leq L||\mathbf{x}-\mathbf{y}||_2 $$


Esta propiedad nos permite medir y verificar la supocisión de que pequeñas variaciones en las entradas, producen también pequeñas variaciones en las salidas. Podemos esperar una estabílidad mínima en los algorimos de optimización.


**Optimización Convexa**

Es un tipo de optimización muy especializada que tiene muchas garantías de convergencia, pero que es restringido a una una familia específica de problemas (e.g. se tienen matrices Hesianas positivas por todos lados, no tienen puntos de silla, el ópitmo local es óptimo global ).

No es el caso de problemas en Aprendizaje Profundo.



https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12


























	









































	

	