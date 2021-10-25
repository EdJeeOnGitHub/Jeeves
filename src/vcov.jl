struct vcovIID <: vcov
end

"""
    inference(N, K, resid, β, XX_inv, vcov)

Perform inference given model primitives. This function doesn't 
use model objects but residuals, βs, and XXs so we can method dispatch 
the appropriate "meat" in the vcv matrix depending on the model passed.

These are defined later:
e.g. TSLS will XX_inv = (X' P_Z X)^{-1} whereas OLS will just pass (X'X)^{-1}
"""
function inference(N::Int,
                   K::Int,
                   resid::Vector, 
                   β::Vector, 
                   XX_inv::Matrix, 
                   X, # not used for IID but required for cluster/hetero
                      # Need a better design pattern here.
                   vcov::vcovIID)
    σ_sq = sum(resid.^2) / (N - K)
    vcov_matrix = XX_inv * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    pval = 2 .* cdf.(TDist(N - K), -abs.(β ./ se))  
    
    return se, pval, σ_sq, vcov_matrix
end


function Ftest(fit_obj::LinearModelFit; signif_level = 0.05)
    resid = fit_obj.modelfit.resid
    RSS_1 = sum((fit_obj.y .- mean(fit_obj.y)).^2)
    RSS_2 = sum(resid.^2)
    K = fit_obj.K
    N = fit_obj.N
    F = ((RSS_1 - RSS_2)/(K - 1)) / ((RSS_2)/(N-K))
    F_CV = quantile(FDist(K - 1, N - K), 1 - signif_level)
    reject = F > F_CV
    return F, F_CV, reject
end

    

# Now we just use method dispatch to call the necessary inference function 
# whilst also handling different ways of passing arguments to inference.

# N.B. generic vcov::vcov. Hopefully we should only need to write a single 
# vcov function  for each type of SE using primitives and everything in 
# this file will dispatch away  
"""
inference taking β and resid as arguments
"""
function inference(resid::Vector,
                   β::Vector, # todo change type to OLS?
                   fit::Union{LinearModel,LinearModelFit}, 
                   vcov::vcov)
    R = fit.R
    N = fit.N
    K = fit.K
    X = fit.X
    XX_inv = inv(cholesky(R' * R))
    return inference(N, K, resid, β, XX_inv, X, vcov) 
end


"""
OLS inference using fitted model to pass arguments to above function
"""
function inference(fit::FittedOLSModel, vcov::vcov)
    resid = fit.modelfit.resid
    β = fit.modelfit.β
    return return inference(resid, β, fit, vcov) 
end


"""
TSLS inference passing X ' P_Z X to meat matrix.
"""
function inference(resid::Vector, 
                   β::Vector, 
                   P_Z::Matrix, 
                   fit::TSLSModel, vcov::vcov)
    X = fit.X 
    XX_inv =  inv(X' * P_Z * X)
    return inference(fit.N, fit.K, resid, β, XX_inv, P_Z*X, vcov)
end
"""
TSLS using fitted model
"""
function inference(fit::FittedTSLSModel, vcov::vcov)
    X = fit.X
    P_Z = fit.P_Z 
    XX_inv =  inv(X' * P_Z * X)
    return inference(fit.N, fit.K, fit.modelfit.resid, fit.modelfit.β, XX_inv, P_Z*X, vcov)
end