A simple framework to define and train Artificial Neural Networks using:
- Stochastic Gradient descent
- BFGS
- L-BFGS

------------------------------------------------------------------------
The project hierarchy is as follows:
- Data/
    MLCup/    contains the ML-Cup datasets
    monk_datasets/    contains the MONK datasets
- src/
    


------------------------------------------------------------------------

Here is the minimal amount of code to create a network and a dataset, and fit
the network on the dataset:


```
from neural_network import *
import numpy as np

# training set
x_train = np.array([ [x] for x in np.linspace(0, 2, 50)])
y_train = np.sin(x_train) + x_train**2 + 2

# define network architecture
architecture = [1, 5, 1]
neurons = [InputNeuron, SigmoidNeuron, LinearNeuron]
net = Network(architecture, neurons)

# fitting
tr_loss, tr_error, _, _, _, gradient_norm = net.train_SGD(x_train, y_train,
                                                             x_test=None,
                                                             y_test=None,
                                                             lossObject=SquaredError(),
                                                             epochs=1000,
                                                             learning_rate=0.01,
                                                             batch_size=16,
                                                             momentum=0.9,
                                                             regularization=0.001,
                                                             epsilon=1e-2)
```

which prints ```stop: norm gradient. Epoch 140```, meaning that the optimization has stopped
after 140 iterations due to the satisfaction of the stopping criterion on the norm of the gradient.
