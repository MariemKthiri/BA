using TensorFlow    
using Distributions

# Network training parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

nAntennas = 8
nBatches = 20

# Network architecture parameters
img_size_x = 28
img_size_y = 28
n_input = img_size_x*img_size_y # MNIST data input (imgage shape: 28*28)
n_classes = 10   # MNIST total classes (0-9 digits)
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features


x = placeholder(Float32, shape=[nAntennas, nBatches])
y = placeholder(Float32, shape=[nAntennas, nBatches])


# We construct layer 1 from a weight set, a bias set and the activiation function used
# to process the impulse set of features for a given example in order to produce a 
# predictive output for that example.
#
#  w_layer_1:    the weights for layer 1.  The first index is the input feature (pixel)
#                and the second index is the node index for the perceptron in the first
#                layer.
#  bias_layer_1: the biases for layer 1.  There is a single bias for each node in the 
#                layer.
#  layer_1:      the activation functions for layer 1
#
#w_layer_1 = Variable(random_normal([nAntennas, nAntennas*nAntennas]))
w_layer_1 = get_variable("weights_layer1", [nAntennas, square(nAntennas)], Float64)
bias_layer_1 = Variable(random_normal([nAntennas]))
# The mnist beginners example uses a softmax activation function; here we will 
# use a sigmoid function instead.
layer_1 = nn.softmax(add(matmul(x,w_layer_1),bias_layer_1))

# We construct layer 2 in an analogous way; now we have a second set of weights,
# biases and activation functions.  These are named similarly to the first set.
# the changes are:
#   1) the input number of features to the second layer, corresponds to the number
#      of nodes in the first
#   2) layer 1 is used in the matrix multiplication when constructing layer 2 
# as opposed to the data placeholder x.
#
#  w_layer_2:    the weights for layer 2.  The first index is the input feature (pixel)
#                and the second index is the node index for the perceptron in the first
#                layer.
#  bias_layer_2: the biases for layer 2.  There is a single bias for each node in the 
#                layer.
#  layer_2:      the activation functions for layer 2
#
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))
# The mnist beginners example uses a softmax activation function; here we will
# use a sigmoid function instead.
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,w),bias_layer_2))

# Similarly we now construct the output of the network, where the output layer
# combines the information down into a space of evidences for the 10 possible
# classes in the problem.
output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
bias_output = tf.Variable(tf.random_normal([n_classes]))
output_layer = tf.matmul(layer_2, output) + bias_output