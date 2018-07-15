mutable struct ConvNN

    inTransform
    outTransform
    learning_rate

    nn_params

    gradient
    predict
    opt_alg

    ConvNN() = new()
end

"""
    ConvNN(nLayers, nIn; inTransform = (x) -> x, outTransform = (x) -> x, learning_rate=1e-3,reg_coeff=1e-6,activation=nn.relu)

Input and output of the NN are complex-valued vectors of dimension `nFilterLength,nCoherence,nBatches`
where `nFilterLength` is calculated from `nIn` and `inTransform`.
This method performs a random initialization of all kernels and biases.
"""
function ConvNN(nLayers::Integer, nIn::Integer;
                inTransform = (x) -> x, outTransform = (x) -> x,
                learning_rate=1e-3, reg_coeff=1e-6, activation=myrelu)

    nFilterLength = length(inTransform(ones(nIn)))
    kernels, biases = init_random(nFilterLength, nLayers)


    
    return ConvNN(kernels, biases, inTransform = inTransform, outTransform = outTransform,
                  learning_rate=learning_rate, reg_coeff=reg_coeff, activation=activation)
end

# This method takes set values as input for kernels and biases
function ConvNN(kernels::Array{Any}, biases::Array{Any};
                inTransform = (x) -> x, outTransform = (x) -> x,
                learning_rate=1e-3, reg_coeff=1e-6, activation=myrelu)
    est = ConvNN()

    est.inTransform   = inTransform
    est.outTransform  = outTransform
    est.learning_rate = learning_rate

    nLayers       = length(kernels)

    nn_params = Dict{Symbol,Any}()
    nn_params[:kernels] = kernels
    nn_params[:biases]  = biases

    est.nn_params = nn_params

    function get_filt(nn_params, x)
        nLayers  = length(nn_params[:kernels])
        nBatches = size(x,3)
        for i in 1:nLayers-1
            x = activation(circ_conv(x, nn_params[:kernels][i] ) .+ nn_params[:biases][i])
        end
        # Last layer is without activation function
        return circ_conv( x, nn_params[:kernels][end] ) .+ nn_params[:biases][end]
    end
    
    # prediction function only estimates transformed channel
    # outTransform has to be applied at a later stage
    function predict(nn_params, y)
        z = inTransform(y)
        (nFilterLength, nCoherence, nBatches) = size(z)

        w = get_filt(nn_params, mean(abs2.(z),2))

        # The output (before transformation) of the NN is an element-wise multiplication
        # of the filter with the (transformed) input data
        return (real(z) .* w, imag(z) .* w)
        # note: multiplication with a complex number will lead AutoGrad's backpropagation
        # algorithm into thinking that w is complex as well and there does not seem to
        # be a nice workaround
    end

    function cost_function(nn_params, y, x0)
        (xhatr,xhati) = predict(nn_params,y)

        nBatches      = size(y,3)
        nFilterLength = size(xhatr,1)

        # get real/imaginary parts of output transform matrix
        QoutR = real(outTransform(eye(nFilterLength)))
        QoutI = imag(outTransform(eye(nFilterLength)))

        # Regularizer
        cost = 0.0
        for i in 1:nLayers
            cost += mean(abs2, nn_params[:kernels][i])
        end
        cost = reg_coeff*cost

        # Loss
        for b in 1:nBatches
            xoutr = QoutR*xhatr[:,:,b] - QoutI*xhati[:,:,b]
            xouti = QoutR*xhati[:,:,b] + QoutI*xhatr[:,:,b]
            x0r   = real(x0[:,:,b])
            x0i   = imag(x0[:,:,b])
            cost += mean(abs2, x0r - xoutr)/nBatches
            cost += mean(abs2, x0i - xouti)/nBatches
        end
        return cost
    end

    est.predict  = predict
    est.gradient = grad(cost_function)
    est.opt_alg  = optimizers(nn_params, () -> Adam(lr=learning_rate))
    est
end

# This method takes a (smaller) network as input and interpolates
# the values of the kernels and biases to the (larger) new filter length.
function resize!( est::ConvNN, nIn::Integer;
    inTransform = (x) -> x, outTransform = (x) -> x, learning_rate=est.learning_rate )
    est.inTransform   = inTransform
    est.outTransform  = outTransform
    nLayers          = length(est.nn_params[:kernels])
    nFilterLength    = length(est.inTransform(ones(nIn)))
    nFilterLengthOld = length(est.nn_params[:kernels][1])
    for i in 1:nLayers
        kernel_old_itp = interpolate(est.nn_params[:kernels][i], BSpline(Quadratic(Line())), OnGrid())
        bias_old_itp   = interpolate(est.nn_params[:biases][i],   BSpline(Quadratic(Line())), OnGrid())

        # normalize kernel such that its energy remains the same
        est.nn_params[:kernels][i] = kernel_old_itp[linspace(1,nFilterLengthOld,nFilterLength)] .* nFilterLengthOld/nFilterLength
        est.nn_params[:biases][i]  = bias_old_itp[  linspace(1,nFilterLengthOld,nFilterLength)]
    end

    # Re-initialize Adam with new step size
    est.opt_alg = optimizers(est.nn_params, () -> Adam(lr=learning_rate))
end

function reset!( est::ConvNN )
    nLayers       = length(est.nn_params[:kernels])
    nFilterLength = length(est.nn_params[:kernels][1])
    est.nn_params[:kernels], est.nn_params[:biases] = init_random( nFilterLength, nLayers )

    # Re-initialize Adam with new step size
    est.opt_alg = optimizers(est.nn_params, () -> Adam(lr=learning_rate))
end

function init_random( nFilterLength, nLayers )
    kernels = Array{Any}(nLayers)
    biases  = Array{Any}(nLayers)
    for i in 1:nLayers
        kernels[i]= rand(TruncatedNormal(0,0.1,-0.3,0.3), nFilterLength)
        biases[i] = 0.1*ones(nFilterLength)
    end
    return kernels, biases
end


function train!(est::CondNormalKnet.ConvNN, y, x0)
    update!(est.nn_params, est.gradient(est.nn_params,y,x0), est.opt_alg)
end

function estimate(est::ConvNN, y; noOutTransform = false)
    (xr,xi) = est.predict(est.nn_params, y)
    if noOutTransform
        return xr+1im*xi
    else
        return est.outTransform(xr+1im*xi)
    end
end
