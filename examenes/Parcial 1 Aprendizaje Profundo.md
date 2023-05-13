
Aprendizaje Automático
Evaluación Unidad 1,2 y 3.
Prof. Dr. Rodrigo López Farías


1. ¿Qué limitaciones tienen los modelos basados en aprendizaje Biológico en la Primera Ola?
2. ¿Qué limitaciones tuvieron los modelos de aprendizaje automático de la Segunda Ola?
3. ¿Cuáles son las principales de característica de los dos tipos principales de aprendizaje automático?
4. Da un ejemplo de una función $f(x)$ que contenga un punto crítico en un mínimo local.
5. Da un ejemplo de una función $f(x)$ que contenga un punto crítico en un máximo local.
6. A partir de la Serie de Taylor,

$$f(x)=\sum^\infty_{n=0} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

Pero truncando hasta la derivada de orden 2. 

$$f(\mathbf{x}) \approx f(\mathbf{x_0})+(\mathbf{x}-\mathbf{x}_0)^T \mathbf{g} + 1/2 (\mathbf{x}-\mathbf{x}_0)^T \mathbf{H} (\mathbf{x}-\mathbf{x}_0)$$

Describir los pasos necesarios para llegar a la solución de la tasa de aprendizaje: 

$$e^* = \frac{g^T}{g^THg}$$


Escribir en pseudocódigo el algoritmo de descenso de gradiente.

Dada la función de prueba:

$$f(x,y) = a (x^2+y^2)-b xy $$

donde $$a=0.26, b = 0.48$$.



Calcular el gradiente $$\nabla f(x,y)^T = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial x}] $$

Y su matriz Hessiana $$H f(\mathbf{x}){i,j} = \frac{\partial ^2}{\partial x_i \partial x_j} f(\mathbf{x})$$

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/16edfd41354f9adbe656c26498348a8bee5d6c7b)

