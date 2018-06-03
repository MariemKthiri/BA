mutable struct ConvNN

    inTransform
    outTransform

    activation

    y    # observation
    x0   # true state

    biases
    kernels
    x     # estimate 

    train_step
    sess

    ConvNN() = new()
end

"""
    ConvNN(nLayers, nIn; inTransform = (x) -> x, outTransform = (x) -> x, learning_rate=1e-3,reg_coeff=1e-6,activation=nn.relu)

Input and output of the NN are complex-valued vectors of dimension `nCoherence, nFilterLength`
where `nFilterLength` is calculated from `nIn` and `inTransform`.
For mini-batch training, these are stacked into three dimensional arrays of dimension
`nBatches, nCoherence, nFilterLength`.
This method performs a random initialization of all kernels and biases.
"""
function ConvNN(nLayers::Integer, nIn::Integer;
                inTransform = (x) -> x, outTransform = (x) -> x,
                learning_rate=1e-3, reg_coeff=1e-6, activation=nn.relu)

    nFilterLength = length(inTransform(ones(nIn)))

    kernels = Array{Any}(nLayers)
    biases  = Array{Any}(nLayers)
    for i in 1:nLayers
        kernels[i]= rand(TruncatedNormal(0,0.1,-0.3,0.3), nFilterLength)
        biases[i] = 0.1*ones(nFilterLength)
    end
    
    return ConvNN(kernels, biases, inTransform = inTransform, outTransform = outTransform,
                  learning_rate=learning_rate, reg_coeff=reg_coeff, activation=activation)
end

# Initialize TF variables with given kernels and biases
function ConvNN(kernels::Array, biases::Array;
                inTransform = (x) -> x, outTransform = (x) -> x,
                learning_rate=1e-3, reg_coeff=1e-6, activation=nn.relu)
    
    nLayers = length(kernels)

    kernels_tf = Array{TensorFlow.Variables.Variable}(nLayers)
    biases_tf  = Array{TensorFlow.Variables.Variable}(nLayers)
    for i in 1:nLayers
        kernels_tf[i] = Variable(Float32.(kernels[i]))
        biases_tf[i]  = Variable(Float32.(biases[i]))
    end
    return ConvNN(kernels_tf, biases_tf, inTransform = inTransform, outTransform = outTransform,
                  learning_rate=learning_rate, reg_coeff=reg_coeff, activation=activation)

end

# Constructor with initialized TF variables
function ConvNN(kernels::Array{TensorFlow.Variables.Variable}, biases::Array{TensorFlow.Variables.Variable};
                inTransform = (x) -> x, outTransform = (x) -> x,
                learning_rate=1e-3, reg_coeff=1e-6, activation=nn.relu)
    est = ConvNN()
    const nBatches = -1 # special value in TensorFlow (nBatches not when building comp. graph)

    est.activation    = activation

    est.inTransform   = inTransform
    est.outTransform  = outTransform

    nLayers       = length(kernels)
    nFilterLength = get(get_shape(kernels[1]).dims[1])

    # y is the transformed input data
    # the transformation is implemented as a Julia calculation and not as
    # a TensorFlow calculation
    est.y  = placeholder(Complex64) # dims: nBatches, nCoherence, nFilterLength
    # x0 is the original (non-transformed) input that shall be estimated
    est.x0 = placeholder(Complex64) # dims: nBatches, nCoherence, nOut

    est.kernels = kernels
    est.biases  = biases

    # Build NN
    intermediates = Array{TensorFlow.Tensor}(nLayers)

    # For each batch b=1,...,nBatches, the input to the network is given as sum_t |y[b,t,:]|^2
    intermediates[1] = reduce_mean(square(real(est.y)) + square(imag(est.y)), axis=[2]) # dims: nBatches, nFilterLength
    for i in 1:nLayers-1
        intermediates[i+1] = activation(intermediates[i]*kernels[i] + biases[i] ) # dims: nBatches, nFilterLength
    end
    # Last layer is without activation function
    filt = (intermediates[end]*kernels[end]) + biases[end] # dims: nBatches, nFilterLength

    # The output (before transformation) of the NN is an element-wise multiplication
    # of the filter with the (transformed) input data
    est.x = reshape(filt, [nBatches, 1, nFilterLength]) .* est.y # dims: nBatches, nCoherence, nFilterLength

    # x_out = Q*x is implemented as x.'*Q.' because first dim. is batch
    # TF needs to know about Q to calculate gradients
    # later, we can access est.x directly and perform the output transform
    # (efficiently) in Julia or whatever

    # target_shape is a bit messy because nCoherence is not known when the
    # NN is defined
    #QoutTP = constant(Complex64.(outTransform(eye(nFilterLength)).'))
    #target_shape = concat([size(est.y)[1:2], size(QoutTP)[2:2]], 1)
    #est.x_out = reshape(reshape(est.x, [-1,nFilterLength]) * QoutTP, target_shape) # dims: nBatches, nCoherence, nOut

    # Set learning parameters
    regu = reduce_mean(square(kernels[1])) # regularization term
    for i in 2:nLayers
        regu = regu + reduce_mean(square(kernels[i]))
    end
    cost_function  = reduce_mean(square(real(est.x-est.x0)) + square(imag(est.x-est.x0))) + (reg_coeff*regu)
    opt_alg        = tf.train.AdamOptimizer(learning_rate)
    est.train_step = tf.train.minimize(opt_alg,cost_function)

    est.sess = Session()
    init = global_variables_initializer()
    run(est.sess,init)

    # close TF-session when network is de-referenced
    # note: All nodes remain in the graph. To reset the graph, restart julia.
    finalizer(est, est -> close(est.sess))
    est
end

cnn_perm = x -> permutedims(x,[3,2,1])
function train!(est::ConvNN, y, x0)
    #run(est.sess, est.train_step, Dict([(est.y, cnn_perm(est.inTransform(y))), (est.x0, cnn_perm(x0))]))
    run(est.sess, est.train_step, Dict([(est.y, cnn_perm(est.inTransform(y))), (est.x0, cnn_perm(x0))]))
end
function estimate(est::ConvNN, y; noOutTransform = false)
    x = cnn_perm(run(est.sess, est.x, Dict([(est.y, cnn_perm(est.inTransform(y)))])))
    if noOutTransform
        return x
    else
        return est.outTransform(x)
    end
end