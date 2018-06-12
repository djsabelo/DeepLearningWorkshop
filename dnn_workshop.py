## Basics
# Load data:
#   - plot X first time window;
#   - load y data and plot it;
#   - check shape

# Define placeholders x and y


# Define weight matrix with normal distribution that gives an output from

# Define bimport tensorflow as tf

## Basics
# Load data:
#   - plot X first time window;
#   - load y data and plot it;
#   - check X and y shapes
#   - Define the number of classes (check "dnn_workshop_data.py")


# Define placeholders:
#       - x with dimensions of  X_train dimensions of 1 channel - dtype=tf.float32
#       - y with dimensions of the number of classes - dtype=tf.int32

# Define a node:
#   - Define the number of desired hidden units - n_units
#   - Define a numpy random matrix with dimensions that fit one of the channels of one window,
#       the result must be a vector of n_units size
#   - Define weight matrix with trainable=True and as initial value the previous matrix
#   - Define bias with initial values of 0 with n_units
#   - Define the first layer (equation of the node) - use matmul

# Define a prediction node

#   - Define a numpy random matrix with dimensions that fit the last layer and gives a vector of n_classes as output
#   - Define the respective bias
#   - Define the logits layer (equation)
#   - Define a softmax layer
#   - Define y_pred (softmax layer gives the most probable class)

# Run example
#   - make y an one_hot_vector matrix
#   - start session and initialize variables
#   - run y_pred for each window and plot result

# plt.plot(predictions)
# plt.show()

## Optimizer - Lets Train ou network!

#   - Define loss with softmax_cross_entropy
#   - Define optimizer and learning rate [0.1-> 0.0001]
#   - Run 100 loops of Optimizer and loss and plot loss (use tf.summary.scalar(variable_name, variable)
#   - Merge all the summaries and write them out to /tmp/mnist_logs (by default)


## LSTM

# Make a double layer LSTM

# Make a classification node

# Make a softmax function

# Make a one-hot-vector

# Train
