**Ejercicios**

1. Escribir un programa en Python que muestre la tabla de verdad del operador lógico ```xor```.
2. Escribir un programa en Python que:   
    a) Pida al usuario el tamaño de una matrix cuadrada,   
    b) Que genere dos matrices A y B de ese mismo tamaño con números positivos aleatorios.  
    c) Que las imprima   
    d) Que devuelva la multiplicación de matrices A*B   
    
   
   
import random

#Ayuda: matrices de numeros enteros generados comprensión de Listas anidadas
#newlist = [expression for item in iterable if condition == True]
#Implementamos la función range() que devuelve una lista de números. Ver la ayuda.

A = [[random.randint(0,10) for j in range(5)] for i in range(5)]
B = [[random.randint(0,10) for j in range(5)] for i in range(5)]  
C = [[None for j in range(5)] for i in range(5)] 


