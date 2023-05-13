#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from math import exp
from random import seed, random
import math
from IPython.display import clear_output

### Construyendo el dataset que queremos ajustar, el famoso problema del xor

dataset = [
    [0,1,1],
    [1,0,1],
    [0,0,0],
    [1,1,0]
    
]



def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    
    return network

# Calcular W^T X + b

def activate(weights, inputs):
    # W^T*X+b
    
    #bias: b
    activation = weights[-1]
    
    #calcular b + W*X
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
        
    return activation

# Transfer neuron activation

def transfer(activation):

    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output

def transfer_derivative(output):
    '''
    Esta es la derivada de la función sigmoide
    d (1/(1+e^-x))/dx = x(1-x)
    '''
    return output * (1.0 - output)

def forward_propagate(network, inputs):
    

    
    #recorremos capa por capa de la red neuronal
    for layer in network:
        
        #preparamos donde guardaremos las entradas de las siguiente capa
        new_inputs = []
        
        for neuron in layer:
            
            #y = W^TX + W_0+b
            activation = activate(neuron['weights'], inputs)
            
            #y = sigm(x)
            neuron['output'] = transfer(activation)
            
            #guardar salida en new_inputs, el tamaño de vector de salida es igual al número de neuronas de la capa actual 
            #y debe coincidir con de la entrada
            
            new_inputs.append(neuron['output'])
        
        # la salida de una capa es la entrada de la siguiente de la red
        inputs = new_inputs
        
    return inputs


# Propaga el error hacia atrás y lo va guardando en la estructura de la red
def backward_propagate_error(network, expected):
    
    #comenzamos con la capa de salida
    for i in reversed(range(len(network))):
        
        # por cada capa de la red asignada a layer
        layer = network[i]
        
        #el tamaño de la capa es equivalente al numero de neuronas
        n_neuronas = len(layer)
        
        #inicializamos error
        errors = list()
        
        # la primera vez calculamos sobre la ultima capa que es a la que tenemos acceso en un inicio
        
        # si la capa i es la última:
        if i == len(network)-1:
            
            # por cada neurona de la capa hacer:
            for j in range(n_neuronas):
                
                #la neurona j de la capa i
                neuron = layer[j]
                
                #el famoso y gorrito que es la estimación de nuestro modelo
                yhat = neuron['output']
                
                #por la codificación, un valor del vector tendra un 1
                y = expected[j]
            
                #esta es la derivada de la función de costo (o error), donde usamos la log verosimilitud y la queremos maximizar
                # Aqui comienza la transferencia
                derivada_error = (yhat-y)/((yhat-1)*yhat)
                errors.append(derivada_error) 
        
        else:
            #aqui son las capas restantes hacia atrás donde propagamos el error
            
            for j in range(n_neuronas):
                error = 0.0
                for neuron in network[i + 1]:
                    
                    #son los pesos de los parametros j de la capa 
                    # multiplicado por la parte de la derivada parcial guardada en delta
                    error += (neuron['weights'][j] * neuron['delta'])
                    
                errors.append(error)
                
        #aqui guardamos todos los delta asociada a cada neurona de cada capa
        for j in range(n_neuronas):
            neuron = layer[j]
            
            #el error magnifica la derivada
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
            
# Update network weights with error
def update_weights(network, row, l_rate):
    
    # por cada capa de la red
    for i in range(len(network)):
        
        #extraer la clasificación real del vector de entrada. 
        inputs = row[:2]
        
        #las salidas ahora son las entradas de la siguiente capa
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
         
        #acceder a las neuronas de la capa i
        for neuron in network[i]:
        
            #actualizar el gradiente 
            
            #primero los pesos W
            for j in range(len(inputs)):
                
                # por cada neurona y cada una de sus entradas
                #neuronDelta tiene parte de la derivada parcial y se completa con inputs[j] que es la derivada 
                #parcial de la funcion \partial W^TX + B / \partial w_j
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            
            #despues actualización del sesgo (bias)
            #es la derivada parcial de la funcion \partial W^TX + B / \partial b_j
            neuron['weights'][-1] += l_rate * neuron['delta']
            
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        
        #por cada vector en el conjunto de entrenamiento.
        for row in train:
            
            # se calcular la salida de un elemento del vector
            zi_e = forward_propagate(network, row)[0]
            
          
            #la posición row[-1] \in {0,1} tendrá un 1 como inicialización.
            #row es unidiemnsional y contiene 0 o 1 que representa la clase y la posición en la c
            # si la etiqueta del data set en y tiene un cero, este será la posicion donde asignaremos un 1
            
            
            zi = int(row[-1])
            
   
            
            
            #aquí el error reportado es la suma log verosimilitud , queremos que sea el máximo
            sum_error += zi*math.log(zi_e)+(1-zi)*math.log(1-zi_e)
        
            #print(expected)
            #propagamos hacia atrás 
            #el error generado por la diferencia de el valor esperado y el valor generado por la red
            # se reajusta la red para predecir mejor el valor esperado
            
     
            backward_propagate_error(network, [zi])
            update_weights(network, row, l_rate)
        clear_output(wait=True)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        

#ejecutamos entrenamiento

network = initialize_network(2,2,1)

X = dataset[0]

zi_e = forward_propagate(network, X)[0]


zi = int(X[-1])

backward_propagate_error(network, [zi])




n_outputs = 1


#train_network(network, dataset, 0.05, 4000, n_outputs)
#for layer in network:
#    print(layer)
    
    
    
for x in dataset:    
    y = forward_propagate(network, x[:2])
    print("expectation:",x[-1],"reality",1*((y[-1])>0.5))
        

            

