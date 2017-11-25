    def BackProp(self):
        #output layer
        tmp = len (self.layers) - 1 
        for i in range (len (self.layers[tmp].neurons)):
            # reference ti current analyzed neuron
            neuron = self.layers[tmp].neurons[i]
            oi = self.layers[tmp-1].nuerons
return
