import tensorflow as tf
import matplotlib.pyplot as plt
from DeepLearningWorkshop.dnn_workshop_data import *

## Basics
# Load data:
#   - plot X first time window;
# plt.plot(X_train[0, :, 0])
# plt.show()

#   - load y data and plot it;

# plt.plot(y_train)
# plt.show()
#   - check X and y shapes
print(np.shape(X_train))
print(np.shape(y_train))
#   - Define the number of classes (check "dnn_workshop_data.py")
n_classes = 6


# Define placeholders:
#       - x with dimensions of  X_train dimensions of 1 channel - dtype=tf.float32
#       - y with dimensions of the number of classes - dtype=tf.int32

x = tf.placeholder(shape=(np.shape(X_train)[1], 1), dtype=tf.float32)
y = tf.placeholder(shape=(n_classes), dtype=tf.int32)

# Define a node:
#   - Define the number of desired hidden units - n_units
n_units = 512
#   - Define a numpy random matrix with dimensions that fit one of the channels of one window,
#       the result must be a vector of n_units size
value = np.random.normal(size=(n_units, np.shape(X_train)[1]))
#   - Define weight matrix with trainable=True and as initial value the previous matrix
W = tf.Variable(initial_value=value, trainable=True, name='W', dtype=tf.float32)
#   - Define bias with initial values of 0 with n_units
b = tf.Variable(initial_value=np.zeros((n_units, 1)), trainable=True, name='b', dtype=tf.float32)
#   - Define the first layer (equation of the node) - use matmul
node = tf.matmul(W, x) + b

# Define a prediction node

#   - Define a numpy random matrix with dimensions that fit the last layer and gives a vector of n_classes as output
V = tf.Variable(initial_value=np.random.normal(size=(n_classes, n_units)), trainable=True, name='V', dtype=tf.float32)
#   - Define the respective bias
c = tf.Variable(initial_value=np.zeros((n_classes, 1)), trainable=True, name='c', dtype=tf.float32)
#   - Define the logits layer (equation)
logits = tf.matmul(V, node) + c
#   - Define a softmax layer
prob = tf.nn.softmax(logits=logits)
#   - Define y_pred (softmax layer gives the most probable class)
y_pred = tf.argmax(prob)

# Run example
#   - make y an one_hot_vector matrix
y_train = np.identity(n_classes)[y_train]

#   - start session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

#   - run y_pred for each window and plot result
predictions = []
for window, label in zip(X_train, y_train):
    predictions.append(session.run(
        y_pred,
        feed_dict={x: np.reshape(window[:, 0], (128, 1)), y: label[0,:]}
    ))

# plt.plot(predictions)
# plt.show()

## Optimizer - Lets Train ou network!

#   - Define loss with softmax_cross_entropy
loss = tf.losses.softmax_cross_entropy(y, logits[:, 0])

#   - Define optimizer and learning rate [0.1-> 0.0001]
learning_rate = 0.01
optimizer = tf.train.RMSPropOptimizer(learning_rate)
optimize_op = optimizer.minimize(loss)

# Run 100 loops of Optimizer and loss and plot loss (use tf.summary.scalar(variable_name, variable)
# tensorboard --logdir=/home/belo/PycharmProjects/DeepLearningWorkshop/train
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train', session.graph)
test_writer = tf.summary.FileWriter('test')
session.run(tf.global_variables_initializer())

predictions = []
for i in range(100):

    indexes = np.random.permutation(np.arange(len(X_train)))
    for window, label in zip(X_train[indexes], y_train[indexes]):
        [sum, loss_history, opt] = session.run(
            [merged, loss, optimize_op],
            feed_dict={x: np.reshape(window[:, 0], (128, 1)), y: label[0,:]}
        )
        train_writer.add_summary(sum, i)
        print(loss_history)




## LSTM

# Make a double layer LSTM

# Make a classification node

# Make a softmax function

# Make a one-hot-vector

# Train

## LSTM optimization

# Experiment k