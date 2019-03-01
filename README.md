A simple framework to define and train Artificial Neural Networks using:
- Stochastic Gradient descent
- BFGS
- L-BFGS

------------------------------------------------------------------------

The project hierarchy is as follows:
- Data/
  - MLCup/    contains the ML-Cup datasets
  - monk_datasets/    contains the MONK datasets
- src/
  - neural_network.py
    - defines the neural network and the methods `train_SGD()`, `train_BFGS` and `train_LBFGS()`.
  - layer.py
    - defines a layer of a neural network.
  - neuron.py
    - defines an abstract neuron of a neural network, and concrete neurons such as `InputNeuron`,
      `LinearNeuron`, `SigmoidNeuron`, `TanHNeuron` and `ReLuNeuron`, each implementing the corresponding
      `activation_function()` and `activation_function_derivative()`.
  - loss_functions.py
    - defines loss functions, and their derivatives, such as `SquaredError()` and `EuclideanError()`
  - grid_search.py
    - defines methods to perform hyperparameters' grid search
  - validation.py
    - defines methods for validation purposes
  - utils.py
    - defines utility methods for plotting curves ecc.

The experiments are performed in the following Python notebooks under `src/`:
- experiments_monk.ipynb
- experiments_cup.ipynb
- convexity_study.ipynb
- bfgs_vs_lbfgs_line_search.ipynb
- bfgs_vs_lbfgs_m.ipynb
- grid_search_example.ipynb
- save_load_network_parameters.ipynb

Unit tests, for a total coverage of 71%, are performed in `test/NeuralNetworkTest.py`:

File | Coverage
-----| --------
layer.py | 100%
loss_functions.py | 97%
neural_network.py | 80%
neuron.py | 84%
utils.py  | 14%
NeuralNetwokTest.py  |  96%

-------------------------------------------------------------------------------

Here is the minimal amount of code to create a network and a dataset, and fit
the network on the dataset:


```python
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
