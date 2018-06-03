"""
This module provides some neural-network-based estimators of the following form:

Input: Vectors y[:,1], ..., y[:,nTrain]

A summary statistic is calculated from the transformed inputs z=Q*y
and this is fed into the NN. The output of the NN is an "element-wise"
filter w, which is then applied to the transformed input:
x[:,t] = w[:] .* z[:,t]
The output of the NN is then given as h[:,t] = Q_out*x[:,t]
"""
module CondNormalKnet

using Knet
using Distributions
using Interpolations

function softmax(x)
    y = exp.(x)
    y = y./sum(y,1)
end

function myrelu(x)
    relu.(x)
end

function circ_conv(x,w)
    nFilterLength = length(w)
    nBatches      = prod(size(x)[2:end])
    xx = reshape(cat(1,zeros(x),x),2nFilterLength,1,1,nBatches)
    ww = reshape(cat(1,w,w),2nFilterLength,1,1,1)
    yy = conv4(xx,ww,padding=(nFilterLength,0))
    res = yy[nFilterLength+2:end, :,:,:]
    return reshape(res,size(x))
end

#include("CondNormalKnet/src/ConvNN.jl")
include("ConvNN.jl")

end
