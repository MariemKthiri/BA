

push!(LOAD_PATH,".")
using DataFrames
using CSV
#using Utils
using Distributions
using StatsBase
#import SCM3GPP; 
const scm = SCM3GPP
#import CondNormalKnet; const cnk = CondNormalKnet
import CondNormalTF; 
const cntf = CondNormalTF
#include("OMP.jl")

#train!(est::cnk.ConvNN, y, x0)  = cnk.train!(est, y, x0)
train!(est::cntf.ConvNN, y, x0) = cntf.train!(est, y, x0)
#estimate(est::cnk.ConvNN, y; noOutTransform = false)  = cnk.estimate(est, y, noOutTransform = noOutTransform)
estimate(est::cntf.ConvNN, y; noOutTransform = false) = cntf.estimate(est, y, noOutTransform = noOutTransform)

#include("sim_helpers.jl")

verbose = true
#-------------------------------------
# Simulation parameters
#
nBatches   = 200
nBatchSize = 50
m = 6   # m : sparsity level
    
#-------------------------------------
# Channel Model
#
snr        = 10 # [dB]
antennas   = [8,16,32,64,96,128]
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=3)

#-------------------------------------

#-------------------------------------

# method that generates "nBatches" channel realizations
#get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)

function get_channel(nAntennas, nCoherence, nBatches, seed)
    srand(seed)
    support = zeros(m,nBatches)
    t = zeros(Complex128, nBatches)
    h = zeros(Complex{Float64}, nAntennas, nCoherence, nBatches)
    I = sample(1:nAntennas,m,replace=false)
    support[:,1] = I
    J = sample(1:nCoherence,m)
    #K = sample(1:nBatches,m,replace=false)
    h[:,:,1] = sparse(I,J,crandn(m),nAntennas,nCoherence)
    t[1] = cov(vec(h[:,:,1]))
    #h = reshape(convert(Array{Float64,2}, h),nAntennas,nCoherence,nBatches)
    for i=2:nBatches
        I = sample(1:nAntennas,m,replace=false)
        support[:,i] = I
        J = sample(1:nCoherence,m)
        #K = sample(1:nBatches,m,replace=false)
        #h = hcat(h,sparse(I,J,crandn(m),nAntennas,nCoherence))     
        h[:,:,i] = sparse(I,J,crandn(m),nAntennas,nCoherence)
        t[i] = cov(vec(h[:,:,i]))
        #if i<=nAntennas t[i,i] = cov(vec(h[:,:,i]))  end
    end
    #h = reshape(convert(Array{Complex{Float64},2}, h),nAntennas,nCoherence,nBatches)
    #return (h,t)
    return (h,support,t)
end
# method that samples C_delta from delta prior
#get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
#get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))

#-------------------------------------
# Learning Algorithm parameters
#

#-------------------------------------
# Learning Algorithm parameters
#
#learning_rates_relu    = 1e-4*64./antennas # make learning rates dependend on nAntennas
#learning_rates_softmax = 1e-3*ones(antennas)
learning_rates_relu    = 1e-6*64./antennas # make learning rates dependend on nAntennas
learning_rates_softmax = 1e-5*ones(antennas)
nLayers = 2
nLearningBatches   = 8000
nLearningBatchSize = 50

results      = DataFrame()
test_results = DataFrame()
supp_results = DataFrame()
nn_est       = Dict{Symbol,Any}()

#for iAntenna in 1:3
    iAntenna = 1
    nAntennas     = antennas[iAntenna]

    verbose && println("Simulating with ", nAntennas, " antennas")

    # Network estimators
    
        # Knet-based networks
        #=nn_est[:KCircReLU] = cnk.ConvNN(nLayers, nAntennas,
                                        inTransform  = (x) -> circ_trans(x,:notransp),
                                        outTransform = (x) -> circ_trans(x,:transp),
                                        learning_rate = learning_rates_relu[iAntenna])
        nn_est[:KToepReLU] = cnk.ConvNN(nLayers, nAntennas,
                                        inTransform  = (x) -> toep_trans(x,:notransp),
                                        outTransform = (x) -> toep_trans(x,:transp),
                                        learning_rate = learning_rates_relu[iAntenna])
        nn_est[:KCircSoftmax] = cnk.ConvNN(nLayers, nAntennas,
                                           inTransform  = (x) -> circ_trans(x,:notransp),
                                           outTransform = (x) -> circ_trans(x,:transp),
                                           learning_rate = learning_rates_relu[iAntenna],
                                           activation = cnk.softmax)
        nn_est[:KToepSoftmax] = cnk.ConvNN(nLayers, nAntennas,
                                           inTransform  = (x) -> toep_trans(x,:notransp),
                                           outTransform = (x) -> toep_trans(x,:transp),
                                           learning_rate = learning_rates_relu[iAntenna],
                                           activation = cnk.softmax)
        =#
        # TensorFlow-based networks
        nn_est[:IdReLU]   = cntf.ConvNN(nLayers, nAntennas,
                                        inTransform  = (x) -> x,
                                        outTransform = (x) -> x,
                                        learning_rate = learning_rates_relu[iAntenna])

        nn_est[:IdSoftmax]   = cntf.ConvNN(nLayers, nAntennas,
                                           inTransform  = (x) -> x,
                                           outTransform = (x) -> x,
                                           learning_rate = learning_rates_softmax[iAntenna],
                                           activation = cntf.nn.softmax)                                           
        


    #for n = 1:nLearningBatches/500
        seed = abs(rand(Int8))
        #train_batch!(nn_est, snr = snr, nBatches = 500*n, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize,seed), verbose = verbose)
        train_batch!(nn_est, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize, seed), verbose = verbose)

        algs = Dict{Symbol,Any}()
        for (alg,nn) in nn_est
            algs[alg] = (y,h,h_cov) -> estimate(nn, y)
        end
        push!(algs, (:GenieAidedMMSE => ((y,supp,h_cov,snr) -> mmse_genie(y,h_cov,snr))))

        (errs,rates) = evaluate(algs, snr = snr, nBatches = nBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nBatchSize, 2), verbose = verbose)

        for alg in keys(algs)
            new_row = DataFrame(MSE        = errs[alg],
                                rate       = rates[alg],
                                Algorithm  = String(alg),
                                #Iteration  = 500*n,
                                SNR        = snr, 
                                nAntennas  = nAntennas,
                                nCoherence = nCoherence)

            if isempty(results)
                results = new_row
            else
                results = vcat(results,new_row)
            end
        end
    #end
    @show results
#end