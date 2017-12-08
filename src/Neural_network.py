from Layer import *
from loss_functions import *

class Network:
    def __init__(self, architecture, neurons):
        check_topology(architecture, neurons)
        self.layers = []
        self.output = [0] * architecture[-1]

        # input layer
        inputNeuron = neurons[0]()
        layer = Layer(architecture[0], 0, inputNeuron)
        self.layers.append(layer)
        # hidden and output layers
        for i in range(1, len(architecture)):
            len_weights = len(self.layers[i-1].neurons)
            neuron = neurons[i](len_weights=len_weights)
            layer = Layer(architecture[i], architecture[i - 1], neuron)
            self.layers.append(layer)

    def forward(self, pattern):
        self.feed_input_neurons(pattern)
        self.propagate_input()
        self.set_output()

    def set_output(self):
        # setting output
        last_layer = self.layers[-1]
        for i in range(len(last_layer.neurons) - 1):  # exclude bias
            self.output[i] = last_layer.neurons[i].getOutput()

    def propagate_input(self):
        # propagate result
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for neuron in layer.neurons[:-1]:  # exclude bias
                weights = neuron.weights
                input_x = self.layers[i - 1].getOutput()
                scores = np.dot(weights, input_x)
                neuron.activation_function(scores)

    def feed_input_neurons(self, data):
        input_layer = self.layers[0]
        for input_neuron, x in zip(input_layer.neurons[:-1], data):  # exclude bias
            input_neuron.activation_function(x)

    def back_propagation(self, target, eta=0.1, momentum=0.9, loss=MisClassified()):
        # 1. get the output vector from the forward step
        output_net = np.array(self.output)

        # propagate the errors backward through the network:

        # 2. for each network output unit compute its error term delta
        delta_output = self.compute_delta_output_units(output_net, target, loss)

        # 3. for each hidden unit compute its error term delta
        delta_vectors = self.compute_delta_hidden_units(delta_output)

        # 4. update network weights
        # array 3d che contiene i cambiamenti da apportare ai pesi, in particolare delta_w[i][j][k] contiene
        # i cambiamenti da apportare nel layer i+1 (no modifiche ad input layer), neurone j, peso k
        delta_w = self.compute_weight_update(delta_vectors, eta)

        # 5 report loss
        loss_value = loss.value(target, output_net)
            
        return delta_w, loss_value

    def compute_weight_update(self, delta_vectors, eta):
        delta_w = []
        for i in range(1, len(self.layers)):
            tmpL = []
            for j in range(len(self.layers[i].neurons) - 1):
                tmpN = []
                for w in range(len(self.layers[i].neurons[j].weights)):
                    # qui errore precedente, ad ogni passo il neuronre di cui si prendere l'output
                    # e diverso, tuo codice aveta ...neurons[j] , adesso ...neuron[w].
                    a = self.layers[i - 1].neurons[w].output
                    b = delta_vectors[-i][j]
                    tmpN.append(eta * self.layers[i - 1].neurons[w].output * delta_vectors[-i][j])
                tmpL.append(tmpN)
            delta_w.append(tmpL)
        return delta_w

    def compute_delta_hidden_units(self, delta_output):
        delta_vectors = [delta_output]
        for hidden_layer_index in range(len(self.layers) - 2, 0, -1):
            delta_layer = []
            for h in range(len(self.layers[hidden_layer_index].neurons)):
                downstream = self.layers[hidden_layer_index + 1].neurons[:-1]
                weights = [neuron.weights[h] for neuron in downstream]
                gradient_flow = np.dot(weights, delta_output)
                d_net = self.layers[hidden_layer_index].neurons[h].activation_function_derivative()
                delta_h = gradient_flow * d_net
                delta_layer.append(delta_h)
            delta_vectors.append(delta_layer)

        return delta_vectors

    def compute_delta_output_units(self, output_net, target, loss):
        output_layer = self.layers[-1]
        af_derivatives = np.array([neuron.activation_function_derivative() for neuron in output_layer.neurons[:-1]])
        diff = loss.derivative(np.array(target), output_net)
        delta_output = np.multiply(af_derivatives, diff)
        return delta_output

    def update_weights(self, delta_w):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k] += delta_w[i - 1][j][k]

    def oldtrain(self, data, targets, epochs, learning_rate,l):  #, batch_size): TODO add batch size
        # fit the data
        losses = []
        for epoch in range(epochs):
            loss_batch = 0
            for pattern, target in zip(data, targets):
                self.forward(pattern)
                delta_w, loss_p = self.back_propagation(target, learning_rate,loss=l)
                loss_batch += loss_p
                self.update_weights(delta_w)
            losses.append(loss_batch)
        return losses

    def sumVector (self,a,b):
        for i in range (0,len(a)):
            for j in range(0,len(a[i])):
                for k in range(0,len(a[i][j])):
                    a[i][j][k] += b[i][j][k]


    def train(self, data, targets, epochs, learning_rate,l,batch_size):
        # fit the data
        losses = []
        delta_wTot = []
        for epoch in range(epochs):
            loss_batch = 0
            delta_wTot = []
            #take only batch_size examples
            for i in range(0,len(data),batch_size):
                delta_wTot = []
                pattern = data[i:i+batch_size]
                target = targets[i:i+batch_size]
                #now really train
                for p,t in zip (pattern,target):
                    self.forward(p)
                    delta_w, loss_p = self.back_propagation(t, learning_rate/batch_size,loss=l)
                    loss_batch += loss_p
                    if delta_wTot == []:
                        delta_wTot=delta_w
                    else:
                        self.sumVector(delta_wTot,delta_w)
                    #update weights
                self.update_weights(delta_wTot)
                #append the total loss in single epoch
            losses.append(loss_batch)
        return losses


    def predict(self, data):
        # predict target variable
        # scores = forward(data)
        pass

    # TODO add output_file as a parameter
    def dump_weights(self):
        # dump neural network weights to file
        # calls serialize on each layer
        for layer in self.layers[1:]:  # exclude input layer
            layer.dump_weights()


def check_topology(architecture, neurons):
    if len(architecture) != len(neurons):
        raise Exception("Architecture miss match")
    if not neurons[0].__name__ is InputNeuron.__name__:
        raise Exception("Input neurons have incorrect type")
    #if not neurons[-1].__name__ is OutputNeuron.__name__:
    #    raise Exception("Output neurons have incorrect type")
    for i in range(1, len(neurons) - 1):
        if neurons[i].__name__ is InputNeuron.__name__ or neurons[i].__name__ is OutputNeuron.__name__:
            raise Exception("Hidden neurons have incorrect type")