using TensorFlow
using Distributions

m = 4
nLayers = 2

function square(x)
    return x^2
end

function vectorize(Xmatrix)
    # get matrix dimensions
    Xmatrix_shape = TensorFlow.get_shape(Xmatrix)
    Xmatrix_rows = get(Xmatrix_shape.dims[1])
    Xmatrix_columns = get(Xmatrix_shape.dims[2])
    @show Xmatrix_rows, Xmatrix_columns

    # reshape matrix = vectorize matrix
    Xvector = reshape(Xmatrix, [Xmatrix_rows*Xmatrix_columns, 1])
    Xvector
end

function devectorize(matrix)
    # get dimensions of vectorized matrix
    Xvector_shape = TensorFlow.get_shape(Xvector)
    Xvector_rows = get(Xvector_shape.dims[1])
    Xvector_columns = get(Xvector_shape.dims[2])
    @show Xvector_rows, Xvector_columns

    # reverse vectorization
    Xmatrix_re = reshape(Xvector, [Xmatrix_rows, Xmatrix_columns])
    Xmatrix_re = TensorFlow.get_shape(Xmatrix_re)
    Xmatrix_re_rows = get(Xmatrix_re.dims[1])
    Xmatrix_re_columns = get(Xmatrix_re.dims[2])
    @show Xmatrix_re_rows, Xmatrix_re_columns
end

function circ_conv(x,w)
    # x,y dims: nBatches, nFilterLength
    const nBatches = -1 # special value in TF
    s     = get_shape(w).dims
    nFilterLength = get(s[1])

    xx = reshape(x,[nBatches,nFilterLength,1,1]) # arrange as 4D array for conv2d
    ww = reshape(tile(w[nFilterLength:-1:1],[2]), [2*nFilterLength,1,1,1]) # reverse, repeat, and arrange as 4D-array for conv2d
    
    z = squeeze(nn.conv2d(xx,ww, strides=[1,1,1,1], padding="SAME"),[3,4]) # dims: nBatches, nFilterLength
end

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

function crandn(dims...)
    sqrt(0.5) * ( randn(dims) + 1im*randn(dims) )
end

sess = Session(Graph())
#hyperparameters
nAntennas = 8
nBatches = 20
snr = 0 #[dB]
#n_input = n_output = [nAntennas, nBatches]
n_hidden = nAntennas
#n_hidden_1 = [n_hidden, nAntennas]
#n_bias_1 = [n_hidden, nBatches]
#n_bias_2 = [nAntennas, nBatches]
#n_hidden_2 = [nAntennas, n_hidden]
#n_output = [3, 2]

#x = rand(Float32, nAntennas, nBatches)
#(xi,_,_) = get_channel(nAntennas, 1, nBatches, abs(rand(Int8)))
#w1 = randn(Float32, nAntennas, nAntennas)
#w1 = ones(Float32, nAntennas, nAntennas)
#e = 10^(-snr/20) * randn(size(xi)...)
#y = w1 * squeeze(xi, 2) + squeeze(e, 2)
# Covariance matrix
#C = y*transpose(y)
#x = zeros(square(nAntennas),1)
#x[:,1] = vectorize(C)

X = placeholder(Float64, shape=[nAntennas, nBatches] )
Y = placeholder(Float64, shape=[nAntennas, nBatches] )
#C = placeholder(Float64, shape=[nAntennas, nAntennas])
#X = get_variable(xi)
#Y = get_variable(y)
#Y_obs = placeholder(Float32)


#W1 = get_variable("weights_layer_1", [n_hidden, square(nAntennas)], Float64)
W1 = get_variable("weights_layer_1", [n_hidden, (nAntennas)], Float64)
b1 = get_variable("bias_layer_1", [n_hidden, 1], Float64)
W2 = get_variable("weights_layer_2", [(nAntennas), n_hidden], Float64)
b2 = get_variable("bias_layer_2", [(nAntennas), 1], Float64)

C = Y*TensorFlow.transpose(Y)
    # get matrix dimensions
    Xmatrix_shape = TensorFlow.shape(C)
    Xmatrix_rows = get(Xmatrix_shape.dims[1])
    Xmatrix_columns = get(Xmatrix_shape.dims[2])
    @show Xmatrix_rows, Xmatrix_columns

    # reshape matrix = vectorize matrix
    Xvector = reshape(C, [Xmatrix_rows*Xmatrix_columns, 1])
#L1 = vectorize(C)
#L1 = (reduce_mean((real(Y))^2 + (imag(Y))^2, axis=[2]))
#L1 = (((real(Y))^2 + (imag(Y))^2))
L2 = nn.softmax(W1*Xvector + b1)
out = (W2*L2 + b2)
    # get dimensions of vectorized matrix
    Xvector_shape = TensorFlow.get_shape(out)
    Xvector_rows = get(Xvector_shape.dims[1])
    Xvector_columns = get(Xvector_shape.dims[2])
    @show Xvector_rows, Xvector_columns

    # reverse vectorization
    Xmatrix_re = reshape(Xvector, [Xmatrix_rows, Xmatrix_columns])
    Xmatrix_re = TensorFlow.get_shape(Xmatrix_re)
    Xmatrix_re_rows = get(Xmatrix_re.dims[1])
    Xmatrix_re_columns = get(Xmatrix_re.dims[2])
    @show Xmatrix_re_rows, Xmatrix_re_columns
hy = Xmatrix_re .* Y
#hy = (out) .* Y
#hy = reshape(out, nAntennas) .* Y
#hy = reshape(out, square(nAntennas)) .* Y

cost = reduce_mean((hy-X)^2)

optimizer = train.AdamOptimizer()
minimize_op = train.minimize(train.GradientDescentOptimizer(.0001), cost)

saver = train.Saver()
# Run training
run(sess, global_variables_initializer())
#run(sess, initialize_all_variables())
checkpoint_path = mktempdir()
info("Checkpoint files saved in $checkpoint_path")
for epoch in 1:1000
    (xi,_,_) = get_channel(nAntennas, 1, 1, abs(rand(Int8)))
    #w1 = randn(Float32, nAntennas, nAntennas)
    w1 = ones(Float32, nAntennas, nAntennas)
    e = 10^(-snr/20) * randn(size(xi)...)
    #y = w1 * squeeze(xi, 2) + squeeze(e, 2)
    y = xi + e
    #for  b in 1:nBatches
        cur_loss, _ = run(sess, [cost, minimize_op], Dict(X=>xi, Y=>y))
        println(@sprintf("Current loss is %.2f.", cur_loss))
        #train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)       
    #end

end