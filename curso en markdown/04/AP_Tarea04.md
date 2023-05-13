# Tarea 4 

###Entrenando Regresión Lineal y Regresión Logística con descenso de gradiente.

1. Dada la función de verosimilitud de la densidad de probabilidad  $$L(\theta) = \sum_{i=1}^N \sqrt{\frac{1}{2\pi\sigma^2}} e^{-\frac{[(\mathbf{w}^T \mathbf{x}_i+b)-y_i]^2}{2\sigma^2}} $$
donde $\theta = (\mathbf{w},b)$

1.1 Calcular su log verosimilitud negativa $NLL(\theta)$ (Que es $NLL(\theta) = C(\theta)$).
1.2 Expresar la función objetivo en terminos de minimización de la $NLL$.  
1.3 Calcular su gradiente $\nabla NLL(\theta)$
1.4 Escribir un programa en Python que que entrene el modelo con el algoritmo de descenso de gradiente dado un conjunto de datos.  

2. Dada la función de verosimilitud de la densidad de probabilidad de Bernoulli: 

$$L(\theta)  = \prod_{i=1}^N sigm(\mathbf{w}^T\mathbf{x}_i)^{y_i} \cdot [1-sigm(\mathbf{w}^T\mathbf{x}_i)]^{(1-{y_i})} $$ 

donde $\theta = (\mathbf{w})$ y $sigm(z)$ es la función sigmoidal. 


2.1 Calcular su log verosimilitud negativa $NLL(\theta)$ (Que es $NLL(\theta) = C(\theta)$).
2.2 Expresar la función objetivo en terminos de minimización de la $NLL$.  
2.3 Calcular su gradiente $\nabla NLL(\theta)$
2.4 Escribir un programa en Python que que entrene el modelo con el algoritmo de descenso de gradiente dado un conjunto de datos. 


3. Encontrar la ecuación del hiperplano $\pi$ que pasa por las coordenadas $P_{0} = [1,0]$ y es la frontera de decisión de un clasificador lineal con pesos $\mathbf{w}=[6,3]$. Graficar la solución en el plano. 

Sabemos que el vector $\mathbf{w}$ es perpendicular al hiperplano que funciona como frontera de decisión:

 $\overrightarrow {P_0P} \cdot \overrightarrow{\mathbf{w}}=0$
