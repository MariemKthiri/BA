push!(LOAD_PATH,".")
using DataFrames
using CSV
using Utils
using Distributions
using StatsBase
import SCM3GPP; const scm = SCM3GPP
#import CondNormalKnet; const cnk = CondNormalKnet
import CondNormalTF; const cntf = CondNormalTF
include("OMP.jl")

#train!(est::cnk.ConvNN, y, x0)  = cnk.train!(est, y, x0)
train!(est::cntf.ConvNN, y, x0) = cntf.train!(est, y, x0)
#estimate(est::cnk.ConvNN, y; noOutTransform = false)  = cnk.estimate(est, y, noOutTransform = noOutTransform)
estimate(est::cntf.ConvNN, y; noOutTransform = false) = cntf.estimate(est, y, noOutTransform = noOutTransform)

include("sim_helpers.jl")

verbose = true
#-------------------------------------
# Simulation parameters
#
nBatches   = 200
nBatchSize = 50
m = 4   # m : sparsity level
sparsity = 4
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
function get_channel2(nAntennas, nCoherence, nBatches, seed, prob)
    b = Bernoulli(prob)
    srand(seed)
    t = zeros(Complex128, nBatches)
    h = zeros(Complex{Float64}, nAntennas, nCoherence, nBatches)
    h[:,:,1] = rand(b, nAntennas, nCoherence)
    t[1] = cov(vec(h[:,:,1]))
    support = len(find(h[:,:,1]))
    for i in 2:nBatches
        h[:,:,i] = rand(b, nAntennas, nCoherence)
        t[i] = cov(vec(h[:,:,i]))
        support = hcat(support, len(find(h[:,:,i])))
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
    #K = sample(1:nBatches,m,replace=false)
    h[:,:,1] = sparse(I,J,crandn(sparsity),nAntennas,nCoherence)
    t[1] = cov(vec(h[:,:,1]))
    #h = reshape(convert(Array{Float64,2}, h),nAntennas,nCoherence,nBatches)
    for i=2:nBatches
        I = sample(1:nAntennas,sparsity,replace=false)
        support[:,i] = I
        J = sample(1:nCoherence,sparsity)
        #K = sample(1:nBatches,m,replace=false)
        #h = hcat(h,sparse(I,J,crandn(m),nAntennas,nCoherence))     
        h[:,:,i] = sparse(I,J,crandn(sparsity),nAntennas,nCoherence)
        t[i] = cov(vec(h[:,:,i]))
        #if i<=nAntennas t[i,i] = cov(vec(h[:,:,i]))  end
    end
    #h = reshape(convert(Array{Complex{Float64},2}, h),nAntennas,nCoherence,nBatches)
    #return (h,t)
    return (h,support,t)
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
nLearningBatches   = 8000
nLearningBatchSize = 50

results      = DataFrame()
test_results = DataFrame()
supp_results = DataFrame()
nn_est       = Dict{Symbol,Any}()

#for snr in -10:5:20
#for sparsity in [2,4,8]
#for iAntenna in 1:length(antennas)
for iAntenna in 1:1
    nAntennas     = antennas[iAntenna]

    verbose && println("Simulating with ", nAntennas, " antennas")

    # Network estimators
    if iAntenna == 1
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
        nn_est[:CircReLU] = cntf.ConvNN(nLayers, nAntennas,
                                        inTransform  = (x) -> circ_trans(x,:notransp),
                                        outTransform = (x) -> circ_trans(x,:transp),
                                        learning_rate = learning_rates_relu[iAntenna])
        nn_est[:ToepReLU] = cntf.ConvNN(nLayers, nAntennas,
                                        inTransform  = (x) -> toep_trans(x,:notransp),
                                        outTransform = (x) -> toep_trans(x,:transp),
                                        learning_rate = learning_rates_relu[iAntenna])
        nn_est[:IdReLU]   = cntf.ConvNN(nLayers, nAntennas,
                                        inTransform  = (x) -> x,
                                        outTransform = (x) -> x,
                                        learning_rate = learning_rates_relu[iAntenna])
        nn_est[:CircSoftmax] = cntf.ConvNN(nLayers, nAntennas,
                                           inTransform  = (x) -> circ_trans(x,:notransp),
                                           outTransform = (x) -> circ_trans(x,:transp),
                                           learning_rate = learning_rates_softmax[iAntenna],
                                           activation = cntf.nn.softmax)
        nn_est[:ToepSoftmax] = cntf.ConvNN(nLayers, nAntennas,
                                           inTransform  = (x) -> toep_trans(x,:notransp),
                                           outTransform = (x) -> toep_trans(x,:transp),
                                           learning_rate = learning_rates_softmax[iAntenna],
                                           activation = cntf.nn.softmax)
        nn_est[:IdSoftmax]   = cntf.ConvNN(nLayers, nAntennas,
                                           inTransform  = (x) -> x,
                                           outTransform = (x) -> x,
                                           learning_rate = learning_rates_softmax[iAntenna],
                                           activation = cntf.nn.softmax)                                           
        
    else
        #=cnk.resize!(nn_est[:KCircReLU],    nAntennas, learning_rate = learning_rates_relu[iAntenna])
        cnk.resize!(nn_est[:KToepReLU],    nAntennas, learning_rate = learning_rates_relu[iAntenna])
        cnk.resize!(nn_est[:KCircSoftmax], nAntennas, learning_rate = learning_rates_relu[iAntenna])
        cnk.resize!(nn_est[:KToepSoftmax], nAntennas, learning_rate = learning_rates_relu[iAntenna])
        =#
        nn_est[:CircReLU]    = cntf.resize(nn_est[:CircReLU], nAntennas,
                                           learning_rate = learning_rates_relu[iAntenna])
        nn_est[:ToepReLU]    = cntf.resize(nn_est[:ToepReLU], nAntennas,
                                           learning_rate = learning_rates_relu[iAntenna])
        nn_est[:IdReLU]      = cntf.resize(nn_est[:IdReLU], nAntennas,
                                           learning_rate = learning_rates_relu[iAntenna])
        nn_est[:CircSoftmax] = cntf.resize(nn_est[:CircSoftmax], nAntennas,
                                           learning_rate = learning_rates_softmax[iAntenna])
        nn_est[:ToepSoftmax] = cntf.resize(nn_est[:ToepSoftmax], nAntennas,
                                           learning_rate = learning_rates_softmax[iAntenna])
        nn_est[:IdSoftmax]   = cntf.resize(nn_est[:IdSoftmax], nAntennas,
                                           learning_rate = learning_rates_softmax[iAntenna])
    end

    #for n = 1:nLearningBatches/500
        seed = abs(rand(Int8))
        #train_batch!(nn_est, snr = snr, nBatches = 500*n, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize,seed), verbose = verbose)
        train_batch!(nn_est, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(sparsity, nAntennas, nCoherence, nLearningBatchSize, seed), verbose = verbose)

        algs = Dict{Symbol,Any}()
        for (alg,nn) in nn_est
            algs[alg] = (y,h,h_cov) -> estimate(nn, y)
        end
        push!(algs, (:GenieAidedMMSE => ((y,supp,h_cov,snr) -> mmse_genie(y,h_cov,snr))))

        (errs,rates) = evaluate(algs, snr = snr, nBatches = nBatches, get_channel = () -> get_channel(sparsity, nAntennas, nCoherence, nBatchSize, 2), verbose = verbose)

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
end
#end
#-------------------------------------
# Comparison with OMP
# N: Number of measurements
MSE_OMP = zeros(length(antennas))
rate_OMP = zeros(length(antennas))
N       = 1000

#for iAntenna in 1:length(antennas)
#for snr in -10:5:20
#for sparsity in [2,4,8]
    rho = 10^(0.1*snr);
for iAntenna in 1:1
    nAntennas  = antennas[iAntenna]
    #nAntennas = iAntenna
    for bb in 1:nBatches
        (h, h_cov, _) = get_channel(sparsity, nAntennas, nCoherence, nBatchSize, 2)
        y = h + 10^(-snr/20) * crandn(size(h)...)
        hest = zeros(Complex128,size(h)...)
        for j=1:nBatchSize ,t=1:nCoherence
                (hest[:,t,j],_) = OMP(y[:,t,j],size(h)[1],m,N)
                
                rate_OMP[iAntenna] += log2(1 + rho*abs2(dot(h[:,t,j],hest[:,t,j]))/max(1e-8,sum(abs2,hest)))/length(h[1,:,:])/nBatches
            #end
        end
        MSE_OMP[iAntenna] += sum(abs2,h-hest)/length(h)/nBatches
    end
    new_row = DataFrame(MSE        = MSE_OMP[iAntenna],
                        rate       = rate_OMP[iAntenna],
                        Algorithm  = "OMP",
                        #Iteration  = 8000,
                        SNR        = snr, 
                        nAntennas  = nAntennas,
                        nCoherence = nCoherence)

    if isempty(results)
        results = new_row
    else
        results = vcat(results,new_row)
    end
    @show results
end
#end
#-------------------------------------
# test OMP
#nBatchSize = 10
#nBatches = 20

#-------------------------------------
# Testing with 128 Antennas
nbr_samples       = 6000
nbr_antennas_test = 8
mean_value_input  = zeros(nbr_antennas_test, nCoherence)
mean_value_output = zeros(nbr_antennas_test, nCoherence,7)
Test_Set_input    = zeros(Complex128, nbr_antennas_test, nCoherence, nbr_samples)
Test_Set_output   = zeros(Complex128, nbr_antennas_test, nCoherence, nbr_samples,7)
htest = zeros(Complex128, nbr_antennas_test, nCoherence, 1)
algs = Dict{Symbol,Any}()
index = Dict{Symbol,Any}()
num = 1
for (alg,nn) in nn_est
    algs[alg] = (y,h,h_cov) -> estimate(nn, y)
    index[alg] = num
    num += 1
end
#To do: influence of nCoherence
for i=1:nbr_samples
    (h, h_cov,_)             = get_channel(sparsity, nbr_antennas_test, nCoherence, 1, 2)
    y = h + 10^(-snr/20) * crandn(size(h)...)
    Test_Set_input[:,:,i]  = squeeze(h, 3)
    #Test_Set_input[:,:,i]  = squeeze(y, 3)
    Test_Set_input[:,:,i]  = sort(abs.(Test_Set_input[:,1,i]), rev=true)
    #mean_value_input += sort(Test_Set_input[:,1,i], by=abs2, rev=true)
    for (alg,est) in algs
        htest = est(y,h,h_cov)
        Test_Set_output[:,:,i,index[alg]] = sort(abs.(htest[:,1,1]), rev=true)
        #mean_value_output[:,:,index[alg]] += Test_Set_output[:,:,i,index[alg]]
    end
    Test_Set_output[:,:,i,7] = sort(abs.(OMP(y[:,1,1],size(h)[1],m,N)[1]), rev=true)
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
                        nCoherence = nCoherence)

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
                        nCoherence = nCoherence)

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
                        nCoherence = nCoherence)

    if isempty(test_results)
        test_results = new_row
    else
        test_results = vcat(test_results,new_row)
    end
end

#-------------------------------------
correct_Support_OMP = 0
supp                = zeros(m,nBatchSize)
supp_est            = zeros(m,nBatchSize)
supp_est_sort       = zeros(m,nBatchSize)
hest                = zeros(Complex128,nbr_antennas_test, nCoherence, nBatchSize)
#for i in 1:1000
    for bb in 1:nBatches
        (h, supp, _) = get_channel(sparsity, nbr_antennas_test, nCoherence, nBatchSize, 2)
        y = h + 10^(-snr/20) * crandn(size(h)...)
        for j=1:nBatchSize
            for t=1:nCoherence
                (hest[:,t,j],supp_est[:,j]) = OMP(y[:,t,j],size(h)[1],m,N)
                #(_,supp_est_sort[:,j]) = sortOMP(y[:,t,j],m)
                correct_Support_OMP += length(intersect(supp[:,j],supp_est[:,j]))/m
            end
        end
        #supp = sort(supp, 1)
        #supp_est = sort(supp_est, 1)
        #if supp == supp_est correct_Support +=1 end
    end
#end
correct_Support_OMP /= (nBatches*nCoherence*nBatchSize)
new_row = DataFrame(correctSupport = correct_Support_OMP,
                    Algorithm       = "OMP",
                    SNR             = snr, 
                    nAntennas       = nbr_antennas_test,
                    nCoherence      = nCoherence)

if isempty(supp_results)
    supp_results = new_row
else
    supp_results = vcat(supp_results,new_row)
end
#-------------------------------------
# test whether NNs results have correct support
correct_Support = Dict{Symbol,Any}()
supp            = zeros(m,nBatchSize)
supp_est        = zeros(m,nBatchSize)
supp_est_sort   = zeros(m,nBatchSize)
hest            = zeros(Complex128,size(h)...)
for (alg,_) in algs
    correct_Support[alg] = 0.0;
end
#for i in 1:1000
    for bb in 1:nBatches
        (h, supp,_) = get_channel(sparsity, nbr_antennas_test, nCoherence, nBatchSize, 2)
        y = h + 10^(-snr/20) * crandn(size(h)...)
        for (alg,est) in algs
            hest = est(y,h,supp)
            for b in 1:nBatchSize
                correct_Support[alg] += length(intersect(sortperm(abs(hest[:,1,b]))[1:m],supp))/m
            end
        end
    end
#end
for (alg,_) in algs
    correct_Support[alg] /= (nBatches*nCoherence*nBatchSize)
    new_row = DataFrame(correctSupport = correct_Support[alg],
                        Algorithm  = String(alg),
                        SNR        = snr, 
                        nAntennas  = nbr_antennas_test,
                        nCoherence = nCoherence)

    if isempty(supp_results)
        supp_results = new_row
    else
        supp_results = vcat(supp_results,new_row)
    end
end
#-------------------------------------------------
CSV.write("/home/mariem/BA/MSE = f(nAntennas) (another copy)/SparseRecovery_sparsity_$(m)_snr_$(snr)_dB_8_antennas.csv", results)

#CSV.write("/home/mariem/BA/Learning Curves/LearningCurves_sparsity_$(m)_snr_$(snr)_dB.csv", results)

CSV.write("/home/mariem/BA/Sparsity Tests (another copy)/test_sparsity_$(m)_snr_$(snr)_dB_8_antennas.csv", test_results)

CSV.write("/home/mariem/BA/Support Tests (another copy)/supp_test_sparsity_$(m)_snr_$(snr)_dB_8_antennas.csv", supp_results)

CSV.write("/home/mariem/BA/MSE = f(SNR)/sparsity_$(m).csv", results)
CSV.write("/home/mariem/BA/MSE = f(sparsity)/snr_$(snr).csv", results)
#end
#-------------------------------------
using Gadfly
mse_plot = plot(results, x=:nAntennas, y=:MSE, color=:Algorithm, Geom.point, Geom.line);
draw(SVG("/home/mariem/BA/MSE = f(nAntennas) (another copy)/MSE_sparsity_$(m)_snr_$(snr)_dB.svg", 3inch, 3inch), mse_plot)

rates_plot = plot(results, x=:nAntennas, y=:rate, color=:Algorithm, Geom.point, Geom.line);
draw(SVG("/home/mariem/BA/MSE = f(nAntennas) (another copy)/rates_sparsity_$(m)_snr_$(snr)_dB.svg", 3inch, 3inch), rates_plot)

#--------------------------------------
for i in antennas[1:3]
    mse_plot = plot(results[results[:nAntennas].==i,[:MSE,:rate,:Algorithm,:Iteration,:SNR,:nAntennas,:nCoherence]], x=:Iteration, y=:MSE, color=:Algorithm, Geom.point, Geom.line,Guide.title("sparsity $(m), SNR = $(snr)dB"));
    draw(SVG("/home/mariem/BA/Learning Curves (another copy)/LearningCurve_$(i)_antennas_sparsity_$(m)_snr_$(snr).svg", 3inch, 3inch), mse_plot)
end

LearningCurve8 = DataFrame()
LearningCurve8 = results[results[:nAntennas].==8,[:MSE,:rate,:Algorithm,:Iteration,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/Learning Curves (another copy)/LearningCurve8_sparsity_$(m)_snr_$(snr).csv", LearningCurve8)

LearningCurve16 = DataFrame()
LearningCurve16 = results[results[:nAntennas].==16,[:MSE,:rate,:Algorithm,:Iteration,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/Learning Curves (another copy)/LearningCurve16_sparsity_$(m)_snr_$(snr).csv", LearningCurve16)

LearningCurve32 = DataFrame()
LearningCurve32 = results[results[:nAntennas].==32,[:MSE,:rate,:Algorithm,:Iteration,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/Learning Curves (another copy)/LearningCurve32_sparsity_$(m)_snr_$(snr).csv", LearningCurve32)

#for (alg,est) in nn_est cntf.save(est, "/home/mariem/BA/$(String(alg))_copy3.jl") end

plot_8_antennas = DataFrame()
plot_8_antennas = results[results[:nAntennas].==8,[:MSE,:rate,:Algorithm,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/MSE = f(SNR)/sparsity_$(m)_8_antennas.csv", plot_8_antennas)

plot_16_antennas = DataFrame()
plot_16_antennas = results[results[:nAntennas].==16,[:MSE,:rate,:Algorithm,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/MSE = f(SNR)/sparsity_$(m)_16_antennas.csv", plot_16_antennas)

plot_32_antennas = DataFrame()
plot_32_antennas = results[results[:nAntennas].==32,[:MSE,:rate,:Algorithm,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/MSE = f(SNR)/sparsity_$(m)_32_antennas.csv", plot_32_antennas)


plot_8_antennas = DataFrame()
plot_8_antennas = results[results[:nAntennas].==8,[:MSE,:rate,:Algorithm,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/MSE = f(sparsity)/snr_$(snr)_8_antennas.csv", plot_8_antennas)

plot_16_antennas = DataFrame()
plot_16_antennas = results[results[:nAntennas].==16,[:MSE,:rate,:Algorithm,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/MSE = f(sparsity)/snr_$(snr)_16_antennas.csv", plot_16_antennas)

plot_32_antennas = DataFrame()
plot_32_antennas = results[results[:nAntennas].==32,[:MSE,:rate,:Algorithm,:SNR,:nAntennas,:nCoherence]]
CSV.write("/home/mariem/BA/MSE = f(sparsity)/snr_$(snr)_32_antennas.csv", plot_32_antennas)
