using TensorFlow

input = placeholder(Float64, shape=[32, 91])
truth = placeholder(Int64, shape=[32])

weights = get_variable("weights_0", [91, 91], Float64)
biases = get_variable("biases_0", [91], Float64)
output = nn.relu(input*weights + biases)

weights = get_variable("weights_1", [91, 46], Float64)
biases = get_variable("biases_1", [46], Float64)
output = nn.relu(output*weights + biases)

weights = get_variable("weights_2", [46, 2], Float64)
biases = get_variable("biases_2", [2], Float64)
output = output*weights + biases

loss = nn.sparse_softmax_cross_entropy_with_logits(output, truth)
minimize = train.minimize(train.GradientDescentOptimizer(1e-4), loss)

sess = Session()
run(sess, initialize_all_variables())

input_data = randn(32, 91)
truth_data = clamp(rand(Int32, 32), 1, 2)
run(sess, minimize, Dict(input => input_data, truth => truth_data))