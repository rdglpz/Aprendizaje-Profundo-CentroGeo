import math
import numpy as np

# create RNN architecture
learning_rate = 0.0001
seq_len = 50
max_epochs = 100
hidden_dim = 100
output_dim = 1
bptt_truncate = 5 # backprop through time --> lasts 5 iterations
min_clip_val = -10
max_clip_val = 10

def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_loss(X, Y, U, V, W):
    loss = 0.0
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
        prev_activation = np.zeros((hidden_dim, 1)) # value of previous activation
        for timestep in range(seq_len):
            new_input = np.zeros(x.shape) # forward pass, done for each step in the sequence
            new_input[timestep] = x[timestep] # define a single input for that timestep
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_activation)
            _sum = mulu + mulw
            activation = sigmoid(_sum)
            mulv = np.dot(V, activation)
            prev_activation = activation
        # calculate and add loss per record
        loss_per_record = float((y - mulv)**2/2)
        loss += loss_per_record
    # calculate loss after first Y pass
    return loss, activation



# takes x values and the weights matrices
# returns layer dictionary, final weights (mulu, mulw, mulv)
def calc_layers(x, U, V, W, hprev):
    
    '''
    Calcula 
    mulw = Whprev, 
    mulu = Ux
    h,hprev
    '''
    
    
    layers = []
    for timestep in range(seq_len):
        new_input = np.zeros(x.shape)
        new_input[timestep] = x[timestep]
        mulu = np.dot(U, new_input)
        mulw = np.dot(W, hprev)
        _sum = mulw + mulu
        h = sigmoid(_sum)
        
        #salida
        yhat = np.dot(V, h)
        layers.append({'activation': h, 'prev_activation': hprev})
        hprev = h

    return layers, mulu, mulw, yhat

def backprop(x, U, V, W, dcdy, mulu, mulw, layers):
    
    
    #derivatives dU, dV, and dW
    dU = np.zeros(U.shape)
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)
    
    #derivatives at time step t 
    #dU_t, dV_t, and dW_t. 
    
    dU_t = np.zeros(U.shape)
    dV_t = np.zeros(V.shape)
    dW_t = np.zeros(W.shape)
    
    
    #derivatives truncated backprop 
    # dU_i, and dW_i
    dU_i = np.zeros(U.shape)
    dW_i = np.zeros(W.shape)
    
    
    #a = Wh + Ux
    a = mulu + mulw
    
    #diff in the last layer
    #dc/dh = dc/dy*dy/dh 
    dsv = np.dot(np.transpose(V), dcdy)
    
    def get_previous_activation_differential(a, ds, W):
        
        # [a * (1 - a)] * [V'] * [y-yhat]
        d_sum = a * (1 - a) * ds
        
        
        dmulw = d_sum * np.ones_like(ds)
        
        #dc/dhprev = dc/dy * dy/dh * dh/da * da/dhprev
        # [W'] * [a * (1 - a)] * [V'] * [y-yhat]
        return np.dot(np.transpose(W), dmulw)
    
    for timestep in range(seq_len):
        dV_t = np.dot(dcdy, np.transpose(layers[timestep]['activation']))
        #ds = dsv
        
        #dc/dhprev = dc/dy * dy/dh * dh/da * da/dhprev
        dprev_activation = get_previous_activation_differential(a, dsv, W)
        
        #derivada truncada a 5 pasos hacia atras
        for ix in range(timestep-1, max(-1, timestep-bptt_truncate-1), -1):
            
            #ds = dc/dh + dc/dhprev
            #
            #dprev_activation = dC/dh 9.7.14
            #dsv: V' dL/dyhat
            #ds = dsv + dprev_activation
            
            
            #dprev_activation = get_previous_activation_differential(a, ds, W)
            
            
            #dc/dw = dc/dh * dh/dw
            #Nota: aqui no se ve que usen la sigmoidal
            dW_i = np.dot(W, layers[timestep]['prev_activation'])
            
            new_input = np.zeros(x.shape)
            new_input[timestep] = x[timestep]
            
            
            dU_i = np.dot(U, new_input)
            
            dU_t += dU_i
            dW_t += dW_i
            
        dU += dU_t
        dV += dV_t
        dW += dW_t
        
        # take care of possible exploding gradients
        if dU.max() > max_clip_val:
            dU[dU > max_clip_val] = max_clip_val
        if dV.max() > max_clip_val:
            dV[dV > max_clip_val] = max_clip_val
        if dW.max() > max_clip_val:
            dW[dW > max_clip_val] = max_clip_val
        
        if dU.min() < min_clip_val:
            dU[dU < min_clip_val] = min_clip_val
        if dV.min() < min_clip_val:
            dV[dV < min_clip_val] = min_clip_val
        if dW.min() < min_clip_val:
            dW[dW < min_clip_val] = min_clip_val
        
    return dU, dV, dW

# training
def train(U, V, W, X, Y, X_validation, Y_validation):
    for epoch in range(max_epochs):
        
        # costo inicial entrenamiento
        loss, prev_activation = calculate_loss(X, Y, U, V, W)

        # costo inicial prueba
        val_loss, _ = calculate_loss(X_validation, Y_validation, U, V, W)
        
        print(f'Epoch: {epoch+1}, Loss: {loss}, Validation Loss: {val_loss}')

        # train model/forward pass
        for i in range(Y.shape[0]):
            
            
            x, y = X[i], Y[i]
            layers = []
            hprev = np.zeros((hidden_dim, 1))
            
            #layers = {'activation': activation, 'prev_activation': prev_activation}
            layers, mulu, mulw, mulv = calc_layers(x, U, V, W, hprev)
                
            # Derivada de la función de costo es dcdy
            dcdy = mulv - y
            dU, dV, dW = backprop(x, U, V, W, dcdy, mulu, mulw, layers)
            
            # update weights
            U -= learning_rate * dU
            V -= learning_rate * dV
            W -= learning_rate * dW
            
    return U, V, W