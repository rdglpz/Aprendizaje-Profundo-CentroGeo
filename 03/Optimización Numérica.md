# 3. Computación numérica



Los algoritmos de Aprendizaje automático resuelven problemas matemáticos a través del tratamiento numérico intensivo.

Esto es que de manera no analítica tratan de resolver problemas de manera iterativa. 
Se requieren repetir muchas veces una operación para encontrar o aproximar una solucion.


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


El cálculo incorrecto se puede dar cuando al menos un elemento de $\mathbf{x}$ es muy grande, generando un $\infty$ que es tratado como un $NaN$ y por lo tanto genera una multiplicación incorrecta en el denominador.

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

Si este número es grande, entonces la salida de  $\mathbf{A}^{-1}{x}$ es muy sensible al error $x$.

Esto es una propiedad matricial con la cual debe de lidiar el cómputo numérico. 



**Optimización basada en gradiente**

Optimización es la tarea de encontrar el valor máximo o mínimo de una función $f(\mathbf{x})$ probando diferetes valores de $\mathbf{x}$.

Optimización está involucrada en TODOS los modelos de aprendizaje automático y profundo. (¿Alguien conoce algún método que no?). 


Usualmente los problemas se definen con respecto a la minimización, y para maximizar se utiliza esta equivalencia.

$$\underset{x}{\text{max }} f(\mathbf{x}) = \underset{\mathbf{x}}{\text{min }} - f(\mathbf{x})$$


La función  $f(\mathbf)$ que queremos optimizar se llama se le llama **función objetivo, criterio, función de costo, función de perdida, función de error**.

A veces nos initeresa mostrar el argumento que minimiza la función.

$$\mathbf{x}^* = \underset{x}{\text{arg min }} f(\mathbf{x})$$

Hay dos tipos de optimización **Convexa y Global**

En optimización **Global** queremos encontrar $f(x) < \forall x \neq x^*$.

En optimización **Convexa** queremos encontrar $f(x) < \{x: \forall x \neq x^* \wedge ||x-x^*|| < \epsilon \}$.

El objetivo de los problemas de optimización es *mover* de la mejor manera $\mathbf{x}$ para encontrar el mínimo dado un problema de optimización.

La expresión general de actualización de $x$ para una función unidimensional
$$x_{t+1} = x_t + \epsilon  \cdot f'(x)) $$

La cual podemos observar 3 componentes:

* El vector $x_t$, (que puede verse como un vector de estado en el tiempo $t$).
* $f'(x))$: El valor de la derivada ue indica cuanto mover a $x$.
* $\epsilon$: Es la tasa de aprendizaje. Es un número de preferencia menor que 1 $f'(x))$.


Idealmente queremos que siempre por cada iteración mejorar la solución previa. $f(x_{t+1})< f(x)$.  Pero no es posible garantizarlo.

Un tipo de optimización muy utilizada en aprendizaje profundo es optimización basada en gradiente.

Esto es porque la funciones objetivo por lo regular son continuas y derivables y se explota esta característica para saber hacia donde mover los parámetros.

$\mathbf{x}$.


**Gradiente**

Porqué es buena idea considerar el gradiente en problemas de optimización?


1. Nos da información sobre la distanca que nos separa del óptimo.
2. Nos da información hacia donde debemos mover la solución actual $x_t$.

El algoritmo de gradiente descendente nos dice que $x_t$ debe moverse en dirección contraria al gradiente o sea, 


$$x_{t+1} = x_t - f'(x)$$

**Puntos críticos o estacionarios**

Son aquellos **mínimos, máximos, puntos de silla** donde la derivada $f'(0) = 0$.
Por lo que :

$$x_{t+1} -   (x_t + \epsilon \cdot f(x_t)) =0 $$

En aprendizaje automático buscaremos una minimización aproximada utilizando optimización local, la cual nos permita al menos obtener un muy buen optimo local.


Optimizaremos funciones de multiples entreadas, una sola salida 

$$f: \mathbf{R}^n \rightarrow \mathbf{R} $$.


**Generalización de la derivada en dimensiones mayores a 1**

cuando $\mathbf{x} \in \mathbf{R}^{n}$ necesitamos calcular las derivadas parciales $\frac{\partial}{\partial x_i} f(\mathbf{x})$de cada entrada $x_i \in \mathbf{x}$. con el objetivo de medir cuando cambia $f$ en funcion de una perturbación en $x_i$. El vector de estas derivadas parciales está dado por:


$$[\frac{\partial f(\mathbf{x})}{\partial x_0}, \frac{\partial f(\mathbf{x})}{\partial x_1},\dots \frac{\partial f(\mathbf{x})}{\partial x_{n-1}}]^T =\nabla_{\mathbf{x}} f(\mathbf{x}) $$


**Algoritmo de descenso mas pronunciado   (Gradiente de descenso)**

Derivación del algoritmo.


Para una sola dimensión es sencillo intuir que la dirección contraria a la derivada es donde se encuentra el descenso, pero para generalizar la idea en una función de mas de una dimensión la dirección que nos da el descenso mas pronunciado se encuentra de la siguiente forma.

Dada una función evaluada en $\mathbf{x}$, como $f(\mathbf{x})  \in R^n $, ¿A qué dirección se encuentra la pendiente mas pronunciada? 

Para contestar esta pregunta, una manera de responder es formulando nuestro problema de una manera adecuada.

$$\underset{\mathbf{u^Tu}=1}{\text{min }} \underset{\alpha \rightarrow 0}{\text{lim }} f(\mathbf{x} + \alpha \mathbf{u})$$

donde 
* $\mathbf{u}$ es el vector unitario que nos indica la dirección a la cual se movería $\mathbf{x}$ 
* $\alpha$ es un valor muy pequeño que tiende a 0.

Primero resolvemos el límite 

$$\underset{\alpha \rightarrow 0}{\text{lim }} f(\mathbf{x} + \alpha \mathbf{u})$$

Con la regla de la cadena hacemos

$$y = f(\mathbf{x}+\alpha \mathbf{u})$$


$$v = \mathbf{x}+\alpha \mathbf{u}$$

$$y = f(\mathbf{v})$$

Resolvemos 

$$\frac{\partial y}{\partial \alpha} = \frac{\partial y}{\partial v}  \frac{\partial v}{\partial \alpha}$$

$$= \nabla f(\mathbf{v}) \cdot \mathbf{u}$$

Reacomodando términos

$$= \mathbf{u}^T \cdot \nabla f(\mathbf{\mathbf{x}+\alpha \mathbf{u}})  $$

Evaluamos para $\alpha =0$

$$= \mathbf{u}^T \cdot \nabla f(\mathbf{\mathbf{x}})  $$

Una vez que resolvimos el límite, el siguiente paso es encontrar la dirección del vector unitario que nos minimice la función :

$$\underset{\mathbf{u},  \mathbf{u}^T \mathbf{u} = 1	 }{\text{min }} = \mathbf{u}^T \cdot \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x}})  $$

Esto es esquivalente a encontrar el ángulo que debe existir entre $\mathbf{u}$ y $ \nabla_{\mathbf{x} } f (\mathbf{x})$

Recordemos que el producto punto entre dos vectores, es igual a multiplicar su magnitud y el coseno del angulo entre ellos.

$$\underset{\mathbf{u},  \mathbf{u}^T \mathbf{u} = 1	 }{\text{min }} =  || \mathbf{u}||_2 || \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x})}|| \text{ cos } \theta $$

Como $\mathbf{u}^T \mathbf{u} =1$, y el gradiente no influye en el proceso de minimización por ser considerado constante. 



$$\underset{\mathbf{u},  \mathbf{u}^T \mathbf{u} = 1	 }{\text{min }} =  \text{ cos } \theta = \frac{\mathbf{u} \cdot \nabla_{\mathbf{x}} f(\mathbf{\mathbf{x})}}{||\mathbf{u}||_2 ||\nabla_{\mathbf{x}} f (\mathbf{x})||_2 }$$

El vector $u$ que minimiza es aquel que genera un ángulo de 180 grados con respecto al vector de las derivadas parciales.

A esto se le llama la **Derivada Direccional**.


Por lo tanto la dirección donde se encuentra la pendiente mas pronunciada es justo en la dirección contraria del vector de derivadas parciales. 


$$x_{t+1} = x_{t} - \epsilon \nabla_\mathbf{x} f (\mathbf{x}) $$

Una estrategia para escoger $\epsilon$ es 

1. escoger un valor muy pequeño
2. Hacer búsueda lineal. ( probar diferentes valores de $\epsilon$ hasta obtener el valor mínim. 

$$\underset{\epsilon}{\text{ min }} f(x_t - \epsilon \nabla_\mathbf{x} f (\mathbf{x}) )$$


Tarea:

Encontrar la dirección de máxima pendiente de la función $$f(\mathbf{x}) = x_1^2 +x_2^2$$ cuando $\mathbf{x} = [1,1]^T$































	

