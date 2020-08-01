# Neural_networks from scratch in Numpy
 Make a Numpy NN for MNIST 

## Problem Statement
We will build a complete neural network using Numpy. We will implement all the steps required to build a network - feedforward, loss computation, backpropagation, weight updates etc.

We will use the MNIST dataset to train your model to classify handwritten digits between 0-9. 

The code is divided into the following sections:

Data preparation
Feedforward
Loss computation
Backpropagation
Parameter updates
Model training and predictions

*FeedForward algorithm:*
![FeedForward](/feedfwd.jpg)

There are some things to take into consideration that will help in implementing the code.

1. The whole data is taken as one batch. No minibatch gradient descent is performed
2. The cumulative input to the layer Z_l is now a step in feedforward
3. The output of the last layer is denoted as H_L instead of P where layer L is the final output layer. Hence, there are L−1 hidden layers.
4. For each layer l, the Z_l is stored as 'activation_memory' and H_(l−1), W_l, b_l are stored as 'linear_memory' to use later in backpropagation

*Loss Function:*
The loss used for multiclass classification is the cross-entropy loss.
loss  = -1*average of the sum of all the elements of the matrix Ylog(HL) multiplied element-wise.
where, HL and Y are matrices.

*Backpropagation algorithm:*
![Backpropagation](/backprop.jpg)

The important points to keep in mind are:

1. The parameters dictionary is getting updated in place at each step.
2. The memories from L_layer_forward consisting of the tuple memory = (linear_memory, activation_memory) for each layer is used in backpropagation
3. The backpropagation process will run in a loop from the last layer to the first, and each loop will compute the gradients for Z,H,W,b.

*Parameter Updates*
We will define the function that updates the weights and biases for all the layers using a 'for' loop using the learning_rate and the gradients stored in gradients.

*Model Training*
This is the final step in which we combine all the functions created above to define an 'L_layer_model'.

After this, we can start the training by specifying the learning rate and the number of iterations.