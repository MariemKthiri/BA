push!(LOAD_PATH,".")
using DataFrames
using CSV
using JLD
using Utils
using Distributions
using StatsBase
import SCM3GPP; const scm = SCM3GPP
import CondNormalKnet; const cnk = CondNormalKnet
include("OMP.jl")


train!(est::CondNormalKnet.ConvNN, y, x0)  = CondNormalKnet.train!(est, y, x0)
estimate(est::CondNormalKnet.ConvNN, y; noOutTransform = false)  = CondNormalKnet.estimate(est, y, noOutTransform = noOutTransform)

include("sim_helpers.jl")

verbose = true
#-------------------------------------
# Simulation parameters
#
nBatches   = 200
nBatchSize = 50
sparsity = 0.25 # 25%
#-------------------------------------
# Channel Model
#
snr        = 0 # [dB]
antennas   = [8,16,32,64,96,128]
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=3)

#-------------------------------------

#-------------------------------------

# method that generates "nBatches" channel realizations
#get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)
function get_channel2(prob, nAntennas, nCoherence, nBatches, seed)
    b = Bernoulli(prob)
    srand(seed)
    t = zeros(Complex128, nBatches)
    h = zeros(Complex{Float64}, nAntennas, nCoherence, nBatches)
    support = Dict{Int64,Any}()
    h[:,:,1] = crandn(nAntennas, nCoherence) .* rand(b, nAntennas, nCoherence)
    t[1] = cov(vec(h[:,:,1]))
    support[1] = (findn(h[:,:,1]))
    for i in 2:nBatches
        h[:,:,i] = crandn(nAntennas, nCoherence) .* rand(b, nAntennas, nCoherence)
        t[i] = cov(vec(h[:,:,i]))
        support[i] =  findn(h[:,:,i])
    end
    return (h,support,t)
end

function get_channel(sparsity, nAntennas, nCoherence, nBatches, seed)
    srand(seed)
    support = zeros(sparsity,nBatches)
    t = zeros(Complex128, nBatches)
    h = zeros(Complex{Float64}, nAntennas, nCoherence, nBatches)
    I = sample(1:nAntennas,sparsity,replace=false)
    support[:,1] = I
    J = sample(1:nCoherence,sparsity)
    h[:,:,1] = sparse(I,J,crandn(sparsity),nAntennas,nCoherence)
    t[1] = cov(vec(h[:,:,1]))
    for i=2:nBatches
        I = sample(1:nAntennas,sparsity,replace=false)
        support[:,i] = I
        J = sample(1:nCoherence,sparsity) 
        h[:,:,i] = sparse(I,J,crandn(sparsity),nAntennas,nCoherence)
        t[i] = cov(vec(h[:,:,i]))
    end
    return (h,support,t)
end

function get_matrix(dims...)
    exp.(im*rand(dims)*2*pi)
end
matrices = Dict{Int,Any}()
for iAntenna in 1:3
    nAntennas = antennas[iAntenna]
    matrices[iAntenna] = Array{Complex128}(50,Int(nAntennas*24/32),nAntennas)
    for i in 1:50
        (matrices[iAntenna][i,:,:] = get_matrix(Int(nAntennas*24/32), nAntennas))
    end
end
function transform(iAntenna, index, x)
    s = size(x)
    if length(s)>2
        return reshape(pinv(matrices[iAntenna][index,:,:])*squeeze(x,2),Int(32/24*s[1]),s[2],s[3])
    else
        return pinv(matrices[iAntenna][index,:,:])*x
    end
end

# method that samples C_delta from delta prior
get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))

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
nLearningBatches   = 6000
nLearningBatchSize = 50

results      = DataFrame()
test_results = DataFrame()
supp_results = DataFrame()
nn_est       = Dict{Symbol,Any}()

# N: Number of measurements
MSE_OMP = zeros(length(antennas))
rate_OMP = zeros(length(antennas))
N       = 300



for iAntenna in 1:3
    nAntennas     = antennas[iAntenna]
    for index in 1:10
        matrix = matrices[iAntenna][index,:,:]
        verbose && println("Simulating with ", nAntennas, " antennas")

        # Network estimators
        if iAntenna == 1
            # Knet-based networks
            nn_est[:KCircReLU] = CondNormalKnet.ConvNN(nLayers, Int(nAntennas*24/32),
                                            inTransform  = (x) -> circ_trans(transform(iAntenna, index, x),:notransp),
                                            outTransform = (x) -> circ_trans(x,:transp),
                                            learning_rate = learning_rates_relu[iAntenna])
            nn_est[:KToepReLU] = CondNormalKnet.ConvNN(nLayers, Int(nAntennas*24/32),
                                            inTransform  = (x) -> toep_trans(transform(iAntenna, index, x),:notransp),
                                            outTransform = (x) -> toep_trans(x,:transp),
                                            learning_rate = learning_rates_relu[iAntenna])
            nn_est[:KIdReLU] = CondNormalKnet.ConvNN(nLayers, Int(nAntennas*24/32),
                                            inTransform  = (x) -> transform(iAntenna, index, x),
                                            outTransform = (x) -> x,
                                            learning_rate = learning_rates_relu[iAntenna])
            nn_est[:KCircSoftmax] = CondNormalKnet.ConvNN(nLayers, Int(nAntennas*24/32),
                                            inTransform  = (x) -> circ_trans(transform(iAntenna, index, x),:notransp),
                                            outTransform = (x) -> circ_trans(x,:transp),
                                            learning_rate = learning_rates_relu[iAntenna],
                                            activation = CondNormalKnet.softmax)
            nn_est[:KToepSoftmax] = CondNormalKnet.ConvNN(nLayers, Int(nAntennas*24/32),
                                            inTransform  = (x) -> toep_trans(transform(iAntenna, index, x),:notransp),
                                            outTransform = (x) -> toep_trans(x,:transp),
                                            learning_rate = learning_rates_relu[iAntenna],
                                            activation = CondNormalKnet.softmax)
            nn_est[:KIdSoftmax] = CondNormalKnet.ConvNN(nLayers, Int(nAntennas*24/32),
                                            inTransform  = (x) -> transform(iAntenna, index, x),
                                            outTransform = (x) -> x,
                                            learning_rate = learning_rates_relu[iAntenna],
                                            activation = CondNormalKnet.softmax)
        else
            CondNormalKnet.resize!(nn_est[:KCircReLU],    Int(nAntennas*24/32), learning_rate = learning_rates_relu[iAntenna])
            CondNormalKnet.resize!(nn_est[:KToepReLU],    Int(nAntennas*24/32), learning_rate = learning_rates_relu[iAntenna])
            CondNormalKnet.resize!(nn_est[:KIdReLU]  ,    Int(nAntennas*24/32), learning_rate = learning_rates_relu[iAntenna])
            CondNormalKnet.resize!(nn_est[:KCircSoftmax], Int(nAntennas*24/32), learning_rate = learning_rates_relu[iAntenna])
            CondNormalKnet.resize!(nn_est[:KToepSoftmax], Int(nAntennas*24/32), learning_rate = learning_rates_relu[iAntenna])
            CondNormalKnet.resize!(nn_est[:KIdSoftmax]  , Int(nAntennas*24/32), learning_rate = learning_rates_relu[iAntenna])       
        end

        #for n = 1:nLearningBatches/500
            seed = abs(rand(Int8))
            #train_batch!(nn_est, snr = snr, nBatches = 500*n, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize,seed), verbose = verbose)
            train_batch!(nn_est, matrix, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel2(sparsity, nAntennas, nCoherence, nLearningBatchSize, seed), verbose = verbose)

            algs = Dict{Symbol,Any}()
            for (alg,nn) in nn_est
                algs[alg] = (y,h,h_cov) -> estimate(nn, y)
            end
            
            (errs,rates) = evaluate(algs, matrix, snr = snr, nBatches = nBatches, get_channel = () -> get_channel2(sparsity, nAntennas, nCoherence, nBatchSize, 2), verbose = verbose)

            for alg in keys(algs)
                new_row = DataFrame(MSE        = errs[alg],
                                    rate       = rates[alg],
                                    Algorithm  = String(alg),
                                    SNR        = snr, 
                                    nAntennas  = nAntennas,
                                    nCoherence = nCoherence,
                                    sparsity   = sparsity)

                if isempty(results)
                    results = new_row
                else
                    results = vcat(results,new_row)
                end
            end
        #end
        @show results
    end
end
#end
#-------------------------------------
# Comparison with OMP

MSE_OMP = zeros(length(antennas))
rate_OMP = zeros(length(antennas))
rho = 10^(0.1*snr);
for iAntenna in 1:3
    nAntennas  = antennas[iAntenna]
    for index in 1:10
        MSE_OMP = zeros(length(antennas))
        rate_OMP = zeros(length(antennas))
        matrix = matrices[iAntenna][index,:,:]
        m = Int(nAntennas*sparsity)
        for bb in 1:nBatches
            (h, h_cov, _) = get_channel2(sparsity, nAntennas, nCoherence, nBatchSize, 2)
            (nAntennas,nCoherence,nBatches) = size(h)
            #(h, h_cov, _) = get_channel(m, nAntennas, nCoherence, nBatchSize, 2)
            y = reshape(matrix * squeeze(h, 2), Int(nAntennas*24/32),nCoherence,nBatches) + 10^(-snr/20) * crandn( Int(nAntennas*24/32),nCoherence,nBatches)
            hest = zeros(Complex128, nAntennas,nCoherence,nBatches)
            for j=1:nBatchSize ,t=1:nCoherence
                    (hest[:,t,j],_) = OMP(y[:,t,j],(size(h)[1]),m,Int(nAntennas*24/32),matrix)
                    
                    rate_OMP[iAntenna] += log2(1 + rho*abs2(dot(h[:,t,j],hest[:,t,j]))/max(1e-8,sum(abs2,hest)))/length(h[1,:,:])/nBatches
                #end
            end
            MSE_OMP[iAntenna] += sum(abs2,h-hest)/length(h)/nBatches
        end
        new_row = DataFrame(MSE        = MSE_OMP[iAntenna],
                            rate       = rate_OMP[iAntenna],
                            Algorithm  = "OMP",
                            SNR        = snr, 
                            nAntennas  = nAntennas,
                            nCoherence = nCoherence,
                            sparsity   = sparsity)

        if isempty(results)
            results = new_row
        else
            results = vcat(results,new_row)
        end
        @show results
    end
end
#end

algs = [:KCircReLU, :KToepReLU, :KIdReLU, :KCircSoftmax, :KToepSoftmax, :KIdSoftmax, :GenieAidedMMSE, :OMP]
final_results = DataFrame()
for iAntenna in 1:3
    nAntennas = antennas[iAntenna]
    errs  = Dict{Symbol,Any}()
    rates = Dict{Symbol,Any}()
    for alg in (algs)
        errs[alg]  = 0.0
        rates[alg] = 0.0
    end
    for alg in (algs)
        results_antenna = results[results[:nAntennas].==nAntennas,[:MSE, :rate, :Algorithm]]
        for i in 1:length(results_antenna[:MSE])
            if String(alg) == String(results_antenna[:Algorithm][i])
                errs[alg]  += results_antenna[:MSE][i]/10
            end
        end
        for i in 1:length(results_antenna[:rate])
            if String(alg) == String(results_antenna[:Algorithm][i])
                rates[alg]  += results_antenna[:rate][i]/10
            end
        end
    end
    for alg in (algs)
        new_row = DataFrame(MSE        = errs[alg],
                            rate       = rates[alg],
                            Algorithm  = String(alg),
                            SNR        = snr, 
                            nAntennas  = nAntennas,
                            nCoherence = nCoherence,
                            sparsity   = sparsity)

        if isempty(final_results)
            final_results = new_row
        else
            final_results = vcat(final_results,new_row)
        end
    end
end

#-------------------------------------
# Testing with 128 Antennas
nbr_samples       = 6000
nbr_antennas_test = 32
mean_value_input  = zeros(nbr_antennas_test, nCoherence)
mean_value_output = zeros(nbr_antennas_test, nCoherence,7)
Test_Set_input    = zeros(Complex128, nbr_antennas_test, nCoherence, nbr_samples)
Test_Set_output   = zeros(Complex128, nbr_antennas_test, nCoherence, nbr_samples,7)
htest = zeros(Complex128, nbr_antennas_test, nCoherence, 1)
algs = Dict{Symbol,Any}()
index = Dict{Symbol,Any}()
num = 1
m = Int(sparsity * nbr_antennas_test)
for (alg,nn) in nn_est
    algs[alg] = (y,h,h_cov) -> estimate(nn, y)
    index[alg] = num
    num += 1
end
#for index in 1:5
    matrix = matrices[3][1,:,:]
    #matrix = eye(nbr_antennas_test)
    #To do: influence of nCoherence
    for i=1:nbr_samples
        (h, h_cov,_)             = get_channel2(sparsity, nbr_antennas_test, nCoherence, 1, 2)
        (nAntennas,nCoherence,nBatches) = size(h)
        y = reshape(matrix * squeeze(h, 2), Int(nAntennas*24/32),nCoherence,nBatches) + 10^(-snr/20) * crandn( Int(nAntennas*24/32),nCoherence,nBatches)
        Test_Set_input[:,:,i]  = squeeze(h, 3)
        #Test_Set_input[:,:,i]  = squeeze(y, 3)
        Test_Set_input[:,:,i]  = sort(abs.(Test_Set_input[:,1,i]), rev=true)
        #mean_value_input += sort(Test_Set_input[:,1,i], by=abs2, rev=true)
        for (alg,est) in algs
            htest = est(y,h,h_cov)
            Test_Set_output[:,:,i,index[alg]] = sort(abs.(htest[:,1,1]), rev=true)
            #mean_value_output[:,:,index[alg]] += Test_Set_output[:,:,i,index[alg]]
        end
        Test_Set_output[:,:,i,7] = sort(abs.(OMP(y[:,1,1],size(h)[1],m,N,matrix)[1]), rev=true)
    end

    mean_value_input = mean(real.(Test_Set_input),[2,3])
    for (alg,est) in algs
        mean_value_output[:,:,index[alg]] = mean(real.(Test_Set_output[:,:,:,index[alg]]),[2,3])
    end
    mean_value_output[:,:,7] = mean(real.(Test_Set_output[:,:,:,7]),[2,3])
    for j=1:nbr_antennas_test
        new_row = DataFrame(input      = j,
                            output     = (mean_value_input[j]),
                            Algorithm  = "NNInput",
                            SNR        = snr, 
                            nAntennas  = nbr_antennas_test,
                            nCoherence = nCoherence,
                            sparsity   = sparsity)

        if isempty(test_results)
            test_results = new_row
        else
            test_results = vcat(test_results,new_row)
        end
    end
    for alg in keys(algs)
        for j=1:nbr_antennas_test
        new_row = DataFrame(input      = j,
                            output     = (mean_value_output[j,:,index[alg]]),
                            Algorithm  = String(alg),
                            SNR        = snr, 
                            nAntennas  = nbr_antennas_test,
                            nCoherence = nCoherence,
                            sparsity   = sparsity)

        if isempty(test_results)
            test_results = new_row
        else
            test_results = vcat(test_results,new_row)
        end
        end
    end
    for j=1:nbr_antennas_test
        new_row = DataFrame(input      = j,
                            output     = (mean_value_output[j,:,7]),
                            Algorithm  = OMP,
                            SNR        = snr, 
                            nAntennas  = nbr_antennas_test,
                            nCoherence = nCoherence,
                            sparsity   = sparsity)

        if isempty(test_results)
            test_results = new_row
        else
            test_results = vcat(test_results,new_row)
        end
    end
#end

#-------------------------------------
correct_Support_OMP = 0
#supp                = zeros(m,nBatchSize)
supp_est            = zeros(Int(m),nBatchSize)
supp_est_sort       = zeros(Int(m),nBatchSize)
hest                = zeros(Complex128,nbr_antennas_test, nCoherence, nBatchSize)
#for index in 1:5
    matrix = matrices[3][1,:,:]
    #matrix = eye(nbr_antennas_test)
    for bb in 1:nBatches
        (h, supp, _) = get_channel2(sparsity, nbr_antennas_test, nCoherence, nBatchSize, 2)
        (nAntennas,nCoherence,nBatches) = size(h)
        y = reshape(matrix * squeeze(h, 2),  Int(nAntennas*24/32),nCoherence,nBatches) + 10^(-snr/20) * crandn( Int(nAntennas*24/32),nCoherence,nBatches)
        for j=1:nBatchSize
            for t=1:nCoherence
                (hest[:,t,j],supp_est[:,j]) = OMP(y[:,t,j],size(h)[1],Int(m),N,matrix)
                #(_,supp_est_sort[:,j]) = sortOMP(y[:,t,j],m)
                correct_Support_OMP += length(intersect(supp[j][1],supp_est[:,j]))/m
            end
        end
        #supp = sort(supp, 1)
        #supp_est = sort(supp_est, 1)
        #if supp == supp_est correct_Support +=1 end
    end
#end
correct_Support_OMP /= (nBatches*nCoherence*nBatchSize)
new_row = DataFrame(correctSupport = correct_Support_OMP,
                    Algorithm      = "OMP",
                    SNR            = snr, 
                    nAntennas      = nbr_antennas_test,
                    nCoherence     = nCoherence,
                    sparsity       = sparsity)

if isempty(supp_results)
    supp_results = new_row
else
    supp_results = vcat(supp_results,new_row)
end

#-------------------------------------
# test whether NNs results have correct support
correct_Support = Dict{Symbol,Any}()
#supp            = zeros(m,nBatchSize)
supp_est        = zeros(Int(m),nBatchSize)
supp_est_sort   = zeros(Int(m),nBatchSize)
#hest            = zeros(Complex128,size(h)...)
for (alg,_) in algs
    correct_Support[alg] = 0.0;
end
#for index in 1:5
    matrix = matrices[3][1,:,:]
    for bb in 1:nBatches
        (h, supp,_) = get_channel2(sparsity, nbr_antennas_test, nCoherence, nBatchSize, 2)
        (nAntennas,nCoherence,nBatches) = size(h)
        y = reshape(matrix * squeeze(h, 2),  Int(nAntennas*24/32),nCoherence,nBatches) + 10^(-snr/20) * crandn( Int(nAntennas*24/32),nCoherence,nBatches)
        for (alg,est) in algs
            hest = est(y,h,supp)
            for b in 1:nBatchSize
                correct_Support[alg] += length(intersect(reverse(sortperm(abs(hest[:,1,b])))[1:Int(m)],supp[b][1]))/m
            end
        end
    end
#end
for (alg,_) in algs
    correct_Support[alg] /= (nBatches*nCoherence*nBatchSize)
    new_row = DataFrame(correctSupport = correct_Support[alg],
                        Algorithm      = String(alg),
                        SNR            = snr, 
                        nAntennas      = nbr_antennas_test,
                        nCoherence     = nCoherence,
                        sparsity       = sparsity)

    if isempty(supp_results)
        supp_results = new_row
    else
        supp_results = vcat(supp_results,new_row)
    end
end

#-------------------------------------------------
CSV.write("SparseRecovery24.csv", results)
CSV.write("SparseRecovery24_final.csv", final_results)
CSV.write("SparseRecovery24_sparsity_test.csv", test_results)
CSV.write("SparseRecovery24_support_test.csv", support_results)