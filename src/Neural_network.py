from src.Layer import *




class Network:
    def __init__(self, architecture, neurons, i, t):
        check_topology(architecture, neurons)
        # input layer
        self.layers = []
        self.input = i
        self.target = t
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

    def forward(self):

        for i in range (len (self.layers[0].neurons)-1):
            # set first layer to input
            neuron = self.layers[0].neurons[i]
            neuron.activation_function( self.input[i] )
        # propagate result
        for i in range (1,len(self.layers)):
            for j in range (len (self.layers[i].neurons) -1 ):  # exclude bias neuron
                neuron = self.layers[i].neurons[j]
                weights = neuron.weights
                input_x = self.layers[i-1].getOutput()
                scores = np.dot (weights , input_x)
                neuron.activation_function(scores)
        # setting output
        last_layer = self.layers[-1]
        for i in range (len (last_layer.neurons) -1 ):
            self.output[i] = last_layer.neurons[i].getOutput()

    def back_propagation(self, target, eta=0.1, momentum=0.9):
        # 1. get the output vector from the forward step
        output_net = np.array(self.output)

        # propagate the errors backward through the network:

        # 2. for each network output unit compute its error term delta
        output_layer = self.layers[-1]
        af_derivatives = np.array([neuron.activation_function_derivative() for neuron in output_layer.neurons[:-1]])
        diff = np.array(target) - output_net
        delta_output = np.multiply(af_derivatives, diff)

        # 3. for each hidden unit compute its error term delta
        delta_vectors = [delta_output]
        hidden_layer_index = 1
        delta_layer = []
        for h in range(len(self.layers[hidden_layer_index].neurons)):
            downstream = self.layers[hidden_layer_index + 1].neurons[:-1]
            weights = [neuron.weights[h] for neuron in downstream]
            gradient_flow = np.dot(weights, delta_output)
            d_net = self.layers[hidden_layer_index].neurons[h].activation_function_derivative()
            delta_h = gradient_flow * d_net
            delta_layer.append(delta_h)
        delta_vectors.append(delta_layer)

            # last_hidden_layer_index = len(self.layers) - 2
            # for i in range(last_hidden_layer_index, -1, -1):
            #     delta_layer = [0] * (len(self.layers[i].neurons) - 1)
            #     for j in range(len(self.layers[i].neurons) - 1):
            #       set of neurons in the next layer (no bias) whose inputs contain the output of current neuron
            # downstream = self.layers[i + 1].neurons[:-1]
            # weights = [neuron.weights[j] for neuron in downstream]
            # gradient_from_next_layer = np.dot(weights, delta_vectors[last_hidden_layer_index - i])
            # derivative = self.layers[i].neurons[j].activation_function_derivative()
            # delta_layer[j] = derivative * gradient_from_next_layer
            # delta_vectors.append(delta_layer)

        # 4. update network weights
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                for w in range(len(self.layers[i].neurons[j].weights)):
                    delta_w = eta * self.layers[i-1].neurons[j].output * delta_vectors[-i][j]
                    self.layers[i].neurons[j].weights[w] += delta_w


    def BackProp(self,eta):
        # using ,as much as possible, the nomenclature used in back propagation lecture
        delta = []       # vectorn in which i save the output neurons' delta (used after)
        outputLayer = len (self.layers) - 1         #output layer
        deltaW = []  # vector in which save what i'll subtract to last layer (initialized = current last layer)

        for i in self.layers[outputLayer].neurons:
            if not isinstance(i, BiasNeuron):
                deltaW.append(copy.deepcopy(i.weights))

        for i in range (len (self.layers[outputLayer].neurons) -1):         #-1 due to exclude bias
            for j in range (len (self.layers[outputLayer].neurons[i].weights)):
                # oi = partial(current's neuron net)/partial current weight
                oi = self.layers[outputLayer-1].neurons[j].output
                # DF = partial(Error)/partial(input to neuron)
                Df = self.output[i] - self.target[i]
                #Dneuron = partial (output's neuron)/partial (current neuron' snet)/ 
                Dneuron = self.layers[outputLayer].neurons[i].activation_function_derivative()
                if j == 0:
                    delta.append(Df*Dneuron)
                    tmp = Df*Dneuron
                else:
                    tmp = Df* Dneuron
                deltaW[i][j] = oi*tmp
        #hidden layers
        for i in range ( outputLayer -1 ,0,-1) :
            for j in range (len(self.layers[i].neurons) -1) : # .1 due to exclude bias
                for k in range (len(self.layers[i].neurons[j].weights)):
                    sum = 0.0   #sum up to k=output layer's number
                    for s in range ( len (self.layers[outputLayer].neurons ) - 1):
                        # tmp = parial(Error)/partial(neruon's net)
                        tmp = delta[s]
                        #Dnet = partial (net)/partial(o)
                        Dnet = self.layers[outputLayer].neurons[s].weights[j]
                        sum += tmp*Dnet
                    # Dout = partial(current neuron's out)/partial (current neuron's net)
                    Dout = self.layers[i].neurons[j].activation_function_derivative()
                    # Dnet = partial (currents neuron's net)/partial(current analyzerd weight's)
                    Dnet = self.layers[i-1].neurons[k].output
                    # update hiddens neuron's weigths
                    self.layers[i].neurons[j].weights[k] -= eta*sum*Dout*Dnet
        # update output neuron's weigths
        for i in range (len(deltaW)):
            for j in range(len(deltaW[i])):
                self.layers[outputLayer].neurons[i].weights[j] -=eta*deltaW[i][j]

    def train(self):
        # fit the data
        # for each iteration/epoch
        #   self.forward()
        #   self.backprop()
        pass

    def predict(self, data):
        # predict target variable
        # scores = forward(data)
        pass

    def serialize(self):
        # dump neural network weights to file
        # calls serialize on each layer
        pass


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