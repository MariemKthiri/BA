using TensorFlow
using Distributions

m = 4
nLayers = 2

function get_channel(nAntennas, nCoherence, nBatches, seed)
    srand(seed)
    support = zeros(m,nBatches)
    t = zeros(Float64, nBatches)
    h = zeros(Float64, nAntennas, nCoherence, nBatches)
    I = sample(1:nAntennas,m,replace=false)
    support[:,1] = I
    J = sample(1:nCoherence,m)
    h[:,:,1] = sparse(I,J,randn(m),nAntennas,nCoherence)
    t[1] = cov(vec(h[:,:,1]))
    for i=2:nBatches
        I = sample(1:nAntennas,m,replace=false)
        support[:,i] = I
        J = sample(1:nCoherence,m)    
        h[:,:,i] = sparse(I,J,rand(m),nAntennas,nCoherence)
        t[i] = cov(vec(h[:,:,i]))
    end
    return (h,support,t)
end

sess = Session(Graph())
#hyperparameters
nAntennas = 8
nBatches = 1
snr = 0 #[dB]
n_hidden = nAntennas

X = placeholder(Float64, shape=[nAntennas, nBatches] )
Y = placeholder(Float64, shape=[nAntennas, nBatches] )

#W1 = get_variable("weights_layer_1", [n_hidden, square(nAntennas)], Float64)
W1 = get_variable("weights_layer_1", [n_hidden, (nAntennas)^2], Float64)
b1 = get_variable("bias_layer_1", [n_hidden, 1], Float64)
W2 = get_variable("weights_layer_2", [(nAntennas)^2, n_hidden], Float64)
b2 = get_variable("bias_layer_2", [(nAntennas)^2, 1], Float64)

C = Y*TensorFlow.transpose(Y)
C = reshape(C, [nAntennas^2, 1])
    # get matrix dimensions
    Xmatrix_shape = TensorFlow.get_shape(C)
    Xmatrix_rows = get(Xmatrix_shape.dims[1])
    Xmatrix_columns = get(Xmatrix_shape.dims[2])
    @show Xmatrix_rows, Xmatrix_columns

    # reshape matrix = vectorize matrix
    Xvector = reshape(C, [Xmatrix_rows*Xmatrix_columns, 1])

L2 = nn.softmax(W1*Xvector + b1)
out = (W2*L2 + b2)
    # get dimensions of vectorized matrix
    Xvector_shape = TensorFlow.get_shape(out)
    Xvector_rows = get(Xvector_shape.dims[1])
    Xvector_columns = get(Xvector_shape.dims[2])
    @show Xvector_rows, Xvector_columns

    # reverse vectorization
    Xmatrix_re = reshape(Xvector, [nAntennas, nAntennas])
    Xmatrix_re_shape = TensorFlow.get_shape(Xmatrix_re)
    Xmatrix_re_rows = get(Xmatrix_re_shape.dims[1])
    Xmatrix_re_columns = get(Xmatrix_re_shape.dims[2])
    @show Xmatrix_re_rows, Xmatrix_re_columns
hy = Xmatrix_re .* Y

cost = reduce_mean((hy-X)^2)

optimizer = train.AdamOptimizer()
minimize_op = train.minimize(train.GradientDescentOptimizer(.0001), cost)

saver = train.Saver()
# Run training
run(sess, global_variables_initializer())
checkpoint_path = mktempdir()
info("Checkpoint files saved in $checkpoint_path")
for epoch in 1:100
    (xi,_,_) = get_channel(nAntennas, 1, nBatches, abs(rand(Int8)))
    #w1 = randn(Float32, nAntennas, nAntennas)
    w1 = ones(Float32, nAntennas, nAntennas)
    e = 10^(-snr/20) * randn(size(xi)...)
    #y = w1 * squeeze(xi, 2) + squeeze(e, 2)
    y = xi + e
    #for  b in 1:nBatches
        cur_loss, _ = run(sess, [cost, minimize_op], Dict(X=>squeeze(xi, 2), Y=>squeeze(y, 2)))
        println(@sprintf("Current loss is %.2f.", cur_loss))
        #train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)       
    #end

end