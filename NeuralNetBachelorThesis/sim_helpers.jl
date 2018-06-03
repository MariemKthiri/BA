# Define "Q" matrix transformations:
# Q*x  = trans(y)
# Q'*x = trans(y,:transp)

# random unitary 
function unitary(n)
    a = randn(n,n)
    f = qrfact(a)
    return f[:Q]
end

# Haar matrix
function haar(n)
    h = [1]
    if n > 2
        h = haar(n/2)
    end
    h_n = kron(h,[1,1])
    h_i = kron(eye(size(h)[1]),[1,-1])
    h = vcat(h_n', h_i')
end

# Q = unitary DFT
function circ_trans(x,tp)
    if tp == :transp
        y = ifft(x,1)*sqrt(size(x,1))
    else
        y = fft(x,1)/sqrt(size(x,1))
    end
    y
end
# Q = first M columns of 2Mx2M unitary DFT
function toep_trans(x,tp)
    if tp == :transp
        y = ifft(x,1)[1:Int(end/2),:,:]*sqrt(size(x,1))
        if length(size(x)) == 2
            y = y[:,:,1]
        end
    else
        y = fft([x;zeros(x)],1)/sqrt(2*size(x,1))
    end
    y
end

function train_batch!(nn_est; snr = 0, nBatches = 1, get_channel = () -> 0.0, verbose = false)
    verbose && @printf "Learning: "
    for b in 1:nBatches
        verbose && mod(b,ceil(Int,nBatches/10))==0 && @printf " ... %.0f%%" b/nBatches*100

        (h,h_cov,_) = get_channel()
        a = squeeze(h,2)
        #a = h[:,:,b]
        y = h + 10^(-snr/20) * crandn(size(h)...)
        #y = reshape(haar(size(a)[1]) * a, size(h)...) + 10^(-snr/20) * crandn(size(h)...)
        #y = reshape((randn(size(a)[1],size(a)[1]) * a), size(h)...) + 10^(-snr/20) * crandn(size(h)...)
        for (_,nn) in nn_est
            train!(nn,y,h)
        end
    end
    verbose && @printf "\n"
end
# Genie-aided MMSE filter uses true covariance matrix
#=mmse_genie(y,h_cov,snr) = begin
    rho = 10^(0.1*snr);
    hest = zeros(y)
    (nAntennas,nCoherence,nBatches) = size(y)
    for b in 1:nBatches
        C = scm.toeplitzHe(h_cov[:,b]) # get full cov matrix
        C = h_cov[:,b]
        Cr = C + eye(nAntennas)./rho
        hest[:,:,b] = C*(Cr\y[:,:,b])
    end
    hest
end=#
mmse_genie(y,supp,h_cov,snr) = begin
    rho = 10^(0.1*snr);
    hest = zeros(y)
    (nAntennas,nCoherence,nBatches) = size(y)
    C = zeros(nAntennas,nAntennas)
    
    for b in 1:nBatches
        for i in supp[b][1]
            C[Int(i),Int(i)] = h_cov[b];
            #C[(i),(i)] = h_cov[b];
        end
        Cr = C + eye(nAntennas)./rho
        hest[:,:,b] = C*(Cr\y[:,:,b])
    end
    hest
end

function evaluate(algs; snr = 0, nBatches = 1, get_channel = () -> 0.0, get_observation = h -> h, verbose = false)
    errs  = Dict{Symbol,Any}()
    rates = Dict{Symbol,Any}()

    rho = 10^(0.1*snr);

    for alg in keys(algs)
        errs[alg]  = 0.0
        rates[alg] = 0.0
    end
    #U = haar(nAntennas)
    # Generate channels, calculate errors and achievable rates
    verbose && @printf "Simulating: "
    for bb in 1:nBatches
        verbose && mod(bb,ceil(Int,nBatches/10))==0 && @printf " ... %.0f%%" bb/nBatches*100        
        (h,supp,h_cov) = get_channel()
        y0  = get_observation(h)
        a = squeeze(y0,2)
        y   = y0 + 10^(-snr/20) * crandn(size(y0)...)
        #y   = reshape((U * a), size(y0)...) + 10^(-snr/20) * crandn(size(y0)...)
        #y   = reshape((rand(size(a)[1],size(a)[1]) * a), size(y0)...) + 10^(-snr/20) * crandn(size(y0)...)
        (nAntennas,nCoherence,nBatchSize) = size(h)
        for (alg,est) in algs
            if alg!= :GenieAidedMMSE
                hest = est(y,h,h_cov)
                errs[alg] += sum(abs2,h-hest)/length(h)/nBatches
                # Achievable rates
                for b in 1:nBatchSize, t in 1:nCoherence
                    if sum(abs2,hest[:,t,b]) < 1e-8
                        verbose && warn(string(alg) * ": channel estimate is zero")
                        continue
                    end
                    
                    rates[alg] += log2(1 + rho*abs2(dot(h[:,t,b],hest[:,t,b]))/max(1e-8,sum(abs2,hest[:,t,b])))/length(h[1,:,:])/nBatches
                end
            else
                hest = mmse_genie(y,supp,h_cov,snr)
                errs[:GenieAidedMMSE] += sum(abs2,h-hest)/length(h)/nBatches
                for b in 1:nBatchSize, t in 1:nCoherence
                    rates[:GenieAidedMMSE] += log2(1 + rho*abs2(dot(h[:,t,b],hest[:,t,b]))/max(1e-8,sum(abs2,hest[:,t,b])))/length(h[1,:,:])/nBatches
                end
            end
        end
        
    end
    verbose && @printf "\n"
    (errs, rates)
end

