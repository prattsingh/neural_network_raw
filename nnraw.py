import numpy as np
from numpy.core.fromnumeric import clip, std
import nnfs
from nnfs.datasets import sine, spiral_data
from nnfs.datasets import sine_data

nnfs.init()

#Create Dataset Code
'''
def spiral_data(points, classes):   #for test dataset
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
    '''

#Dense Layer Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, 
                weight_regularizer_l1 = 0, bias_regularizer_l1 = 0, 
                weight_regularizer_l2 = 0, bias_regularizer_l2 = 0):
        
        #Weight in Inputs(col) X Neurons shape has weight for each col of data, 
        # dot with input(rowsxcol) returns points(rows) for each neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #one bias for each neurons
        self.biases = np.zeros((1, n_neurons))
        
        #Set regulatizer strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs):
        #remember inputs for gradients
        self.inputs = inputs
        #dense layer func (z = wx + b)
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backwards(self, dvalues):
        #gradient wrt weight (dz/dw = x/inputs)
        self.dweights = np.dot(self.inputs.T, dvalues) #output in shape of weights
        #gradient wrt biases (dz/db = 1)
        self.dbiases = np.sum(dvalues , axis = 0 , keepdims=True) #output in shape of biases

        #gradients on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases 

        #gradients wrt inputs
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        #calc output values from inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        #copy dvalues as dinputs for ease of calc 
        self.dinputs = dvalues.copy()
        #gradient of relu wrt z (derivative of max func)
        self.dinputs[self.inputs <= 0] = 0 # 0 where negative else x
    
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs

        #exponential activation func(probabilities)
        exp_values = np.exp(inputs - np.max(inputs , axis=1, keepdims=True))
        
        #normalized prob
        probabilities = exp_values / np.sum(exp_values , axis=1, keepdims=True)
        
        self.output = probabilities

    def backward(self, dvalues):

        #empty array like dvalus
        self.dinputs = np.empty_like(dvalues)

        #gradient of exponential wrt z
        for index , (single_outputs, single_dvalues) \
            in enumerate(zip(self.output, dvalues)):

            single_outputs = single_outputs.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_outputs) - \
                np.dot(single_outputs, single_outputs.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):

        self.dinputs = dvalues * (1 - self.output) * self.output 

class Activation_Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss:
    def regularizer_loss(self, layer):

        #intialize reg loss as 0
        regularization_loss = 0

        #L1 loss - weight
        if layer.weight_regularizer_l1 > 0 :
            regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))

        #l2 loss - weight
        if layer.weight_regularizer_l2 > 0 :
            regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

        #l1 loss - bias
        if layer.bias_regularizer_l1 > 0 :
            regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

        #l2 loss - bias
        if layer.bias_regularizer_l2 > 0 :
            regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases*layer.biases)        

        return regularization_loss

    # clac data loss
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CatagoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        #no of samples in a batch
        samples = len(y_pred)
        #clip data to avoid devision by 0 
        #clip both sides to not drag the mean
        y_pred_clipped = clip(y_pred , 1e-7, 1-1e-7)

        #predicted probabilities
        #only if y is catagorical
        if len(y_true.shape) == 1 :
            correct_confidences = y_pred_clipped[range(samples), y_true]

        #if y is one hot encoded
        elif len(y_true.shape) == 2 :
            correct_confidences = np.sum(y_pred_clipped*y_true , axis=1)
        
        # - log loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        #no of samples in a batch
        samples = len(dvalues)
        #No of labels in each sample
        labels = len(dvalues[0])

        if len(y_true.shape) == 1: #convert sparse into one-hot encoded
            y_true = np.eye(labels)[y_true]

        #gradients of loss wrt softmax activation
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + 
                        (1 - y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses , axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1- 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - 
                        (1 - y_true) / (1 - clipped_dvalues)) / outputs

        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss):
    
    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs

        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs

        self.dinputs = self.dinputs / outputs

class Activation_Softmax_Loss_catagoricalCrossentropy:

        def __init__(self):    
            self.activation = Activation_Softmax()
            self.loss = Loss_CatagoricalCrossentropy()

        def forward(self, inputs, y_true):

            self.activation.forward(inputs)
            self.output = self.activation.output

            return self.loss.calculate(self.output, y_true)

        def backward(self, dvalues, y_true):

            samples = len(dvalues)

            if len(y_true.shape) == 2 : #convert one-hot encoded into sparse
                y_true = np.argmax(y_true, axis= 1)

            self.dinputs = dvalues.copy()

            #combined gradient of loss and softmax wrt to z
            self.dinputs[range(samples), y_true] -= 1

            #normalize gradients
            self.dinputs = self.dinputs / samples


class Optimizer_SGD:

    #Initialize optimizer 
    #lr = 1 for this optimizer
    def __init__(self, learning_rate = 1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    #Call once before parameter update
    def pre_update_params(self):
        #learning rate decay with each iteration
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    #Update parameters
    def update_params(self, layer):

        #if Momentum is set
        if self.momentum:

            #Initializer weight , bias momentums filled with zeroes if layer doesnt have them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)

            #Build weight updates with momentum - take 
            #previous updates * multiplied by retain factor(momentum) and update with current gradients(add -lr*dw)
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            #Set update as momentum for next update
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        #Update current gradients with learning rate without momentum
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        #updated weights/biases for next iteration
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
    
class Optimizer_RMSProp:

    #Initialize settings 
    #default lr = 0.001, epsilon = 1e-7, rho=0.9
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7,
                rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        
        #Create cache filled with zero if layer doesnt have it 
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)

            layer.bias_cache = np.zeros_like(layer.biases)

        #update cache with current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2

        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization 
        # with square rooted cache
        layer.weights += - self.current_learning_rate * \
            layer.dweights / \
                (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * \
            layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:

    #initialize optimizeer , set setting
    #default lr=0.001, epsilon=1e-7, b1 = 0.9, b2=0.999
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7,
                beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        #If layer doesnot have cache arrays,
        # #create them filled with zeroes            
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)

        #Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights

        layer.bias_momentums = self.beta_2 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        #use momentums to normalize weights/biases
        #self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1**(self.iterations + 1)) 
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1**(self.iterations + 1))

        #update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        #corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2**(self.iterations + 1))

        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2**(self.iterations + 1))

        #Update weight/biases 
        #Vanilla SGD + Normalization with square root cache
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
                (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
                (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class dropout:

    def __init__(self, rate):
        #set dropout rate
        self.rate = 1 - rate

    def forward(self, inputs):
        #save inputs
        self.inputs = inputs
        #generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape)/self.rate
        #generate outputs
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        #gradient on values
        self.dinputs = dvalues * self.binary_mask

def Accuracy(softmax_output, y_true):
    predictions = np.argmax(softmax_output, axis=1)
    if len(y_true.shape)== 2:
        y_true = np.argmax(y_true, axis=1)
    
    accuracy = np.mean(predictions == y_true)
    return accuracy

#X,y = spiral_data(1000,3) #for softmax classifier

#X,y = spiral_data(100, 2) #for binary classifier

X,y = sine_data()

#  y = y.reshape(-1, 1) #for binary classifier

dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLu()

#dropout1 = dropout(0.1)

dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLu() 

#activation2 = Activation_Sigmoid() #binary

#loss_activation = Activation_Softmax_Loss_catagoricalCrossentropy()

dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()

#loss_function = Loss_BinaryCrossentropy() #binary

loss_function = Loss_MeanSquaredError()

optimizer = Optimizer_Adam(learning_rate=0.005 ,decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking # how many values have a difference to their ground truth equivalent # less than given precision
# We'll calculate this precision as a fraction of standard deviation # of all the ground truth values
accuracy_precision = np.std(y) / 250

'''
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
#activation2.forward(dense2.output)

#loss_function = Loss_CatagoricalCrossentropy()
loss_activation = Activation_Softmax_Loss_catagoricalCrossentropy()
loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])
print('Loss:', loss)
print('Accuracy:', Accuracy(loss_activation.output, y))

loss_activation.backward(loss_activation.output, y)
dense2.backwards(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backwards(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

'''
for epoch in range(10001):

    #forward pass layer 1
    dense1.forward(X)
    activation1.forward(dense1.output)

    #dropout1.forward(activation1.output)

    #forward pass layer 2
    dense2.forward(activation1.output)

    activation2.forward(dense2.output) #binary/linear

    dense3.forward(activation2.output)

    activation3.forward(dense3.output)
    
    #data_loss = loss_activation.forward(dense2.output, y)

    data_loss = loss_function.calculate(activation3.output, y) #binary/linear

    #regularization_loss = loss_activation.loss.regularizer_loss(dense1) + \
        #loss_activation.loss.regularizer_loss(dense2)

    regularization_loss = loss_function.regularizer_loss(dense1) + \
        loss_function.regularizer_loss(dense2) + \
            loss_function.regularizer_loss(dense3) #binary/linear

    loss = data_loss + regularization_loss

    #accuracy = Accuracy(loss_activation.output, y)

    #predictions = (activation2.output > 0.5) * 1 #binary
    #accuracy = np.mean(predictions == y) #binary

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.abs(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, '+
        f'loss: {loss:.3f} (' + 
        f'data_loss: {data_loss:.3f}, '+ 
        f'reg_loss: {regularization_loss:.3f} ), '+
        f'lr: {optimizer.current_learning_rate}')

    #backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backwards(activation3.dinputs)
    #loss_activation.backward(loss_activation.output, y)
    #dense2.backwards(loss_activation.dinputs)
    activation2.backward(dense3.dinputs) #binary
    dense2.backwards(activation2.dinputs) #binary
    activation1.backward(dense2.dinputs)
    dense1.backwards(activation1.dinputs)

    #optimization
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

#validation set
#x_test, y_test = spiral_data(100, 3)
#x_test, y_test = spiral_data(100, 2)
x_test, y_test = sine_data()

#y_test = y_test.reshape(-1, 1) #binary


dense1.forward(x_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

#loss = loss_activation.forward(dense2.output, y_test)
activation2.forward(dense2.output) #binary

dense3.forward(activation2.output)
activation3.forward(dense3.output)
#loss = loss_function.calculate(activation2.output, y_test) #binary

import matplotlib.pyplot as plt

plt.plot(x_test, y_test)
plt.plot(x_test, activation3.output)
plt.show()

predictions = (activation2.output > 0.5) *1
acc = np.mean(predictions == y_test)

#acc = Accuracy(loss_activation.output, y_test)

#print(f'validation acc:  {acc:.3f}, loss: {loss:.3f}')e






