
Aprendizaje Profundo
Evaluación Unidad 1-3.
CentroGeo - Maestría en Ciencias de Información Geoespacial
Prof. Dr. Rodrigo López Farías
Alumno:  




1. ¿Qué limitaciones tuvieron los modelos de aprendizaje automático de la Segunda Ola? (10/100)

2. ¿Cuáles son las principales características de los dos tipos principales de aprendizaje automático? (10/100)

3. Da un ejemplo de una función $f(x)$ que tenga un punto crítico en un máximo local. Describe los pasos generales para identificarlo. (10/100)

4. A partir de la Serie de Taylor: (20/100)

$$f(x)=\sum^\infty_{n=0} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

Pero truncando hasta la derivada de orden 2, 

$$f(\mathbf{x}) \approx f(\mathbf{x_0})+(\mathbf{x}-\mathbf{x}_0)^T \mathbf{g} + 1/2 (\mathbf{x}-\mathbf{x}_0)^T \mathbf{H} (\mathbf{x}-\mathbf{x}_0)$$

donde:
 
$\mathbf{g}$ : Es el gradiente evaluado en $\mathbf{x}_0$.   
$\mathbf{H}$: El Hessiano evaluado en $\mathbf{x}_0$. 

y

$\mathbf{x}$  = $\mathbf{x_0}-\epsilon \mathbf{g} $

describir los pasos necesarios para llegar a la solución de la tasa de aprendizaje: 


$$\epsilon^* = \frac{\mathbf{g}^T}{\mathbf{g}^T \mathbf{H} \mathbf{g}}$$

5 . Escribir en pseudocódigo el algoritmo de descenso de gradiente (15/100).

Dada la función de prueba:

$$f(x,y) =  ax^2+y(ay-bx)$$

donde $$a=0.26, b = 0.48$$.



Calcular el gradiente. (15/100) $$\nabla f(x,y)^T = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial x}] $$

Y su matriz Hessiana  (20/100)$$H f(\mathbf{x}){i,j} = \frac{\partial ^2}{\partial x_i \partial x_j} f(\mathbf{x})$$

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/16edfd41354f9adbe656c26498348a8bee5d6c7b)

