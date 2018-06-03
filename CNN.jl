using Distributions
using TensorFlow

function vectorize(matrix)
    dim = size(matrix)[1]
    result = matrix[:,1]
    for i in 2:dim
        append!(result,matrix[:,i])
    end
    result
end

function square(x)
    return x*x
end

sess = Session(Graph())
@tf begin
    X = placeholder(Float32, shape=[-1, 28*28])
    Y = placeholder(Float32, shape=[-1, 10])

    W1 = get_variable([28*28, 1024], Float32)
    b1 = get_variable([1024], Float32)
    Z1 = nn.sigmoid(X*W1 + b1)

    W2 = get_variable([1024, 10], Float32)
    b2 = get_variable([10], Float32)
    Z2 = Z1*W2 + b2 # Affine layer on its own, to get the unscaled logits
    Y_probs = nn.softmax(Z2)

    losses = nn.softmax_cross_entropy_with_logits(;logits=Z2, labels=Y) #This loss function takes the unscaled logits
    loss = reduce_mean(losses)
    optimizer = train.minimize(train.AdamOptimizer(), loss)
end



traindata = (flatten_images(MNIST.traintensor()), onehot_encode_labels(MNIST.trainlabels()))
run(sess, global_variables_initializer())


basic_train_loss = Float64[]
@showprogress for epoch in 1:100
    epoch_loss = Float64[]
    for (batch_x, batch_y) in eachbatch(traindata, 1000, (ObsDim.Last(), ObsDim.First()))
        loss_o, _ = run(sess, (loss, optimizer), Dict(X=>batch_x', Y=>batch_y))
        push!(epoch_loss, loss_o)
    end
    push!(basic_train_loss, mean(epoch_loss))
    #println("Epoch $epoch: $(train_loss[end])")
end

