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
                   vcov::vcovIID)
    σ_sq = sum(resid.^2) / (N - K)
    vcov_matrix = XX_inv * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    pval = 2 .* cdf.(TDist(N - K), -abs.(β ./ se))   
    return se, pval, σ_sq, vcov_matrix
end

"""
inference taking β and resid as arguments
"""
function inference(resid::Vector,
                   β::Vector, 
                   fit::Union{LinearModel,LinearModelFit}, 
                   vcov::vcovIID)
    R = fit.R
    N = fit.N
    K = fit.K
    XX_inv = inv(cholesky(R' * R))
    return inference(N, K, resid, β, XX_inv, vcov) 
end


"""
OLS inference using fitted model to pass arguments to above function
"""
function inference(fit::FittedOLSModel, vcov::vcovIID)
    resid = fit.modelfit.resid
    β = fit.modelfit.β
    return return inference(resid, β, fit, vcov) 
end


"""
TSLS inference passing X ' P_Z X to meat matrix.
"""
function inference(resid::Vector, β::Vector, P_Z::Matrix, fit::TSLSModel, vcov::vcovIID)
    X = fit.X 
    XX_inv =  inv(X' * P_Z * X)
    return inference(fit.N, fit.K, resid, β, XX_inv, vcov)
end
"""
TSLS using fitted model
"""
function inference(fit::FittedTSLSModel, vcov::vcovIID)
    X = fit.X
    P_Z = fit.P_Z 
    XX_inv =  inv(X' * P_Z * X)
    return inference(fit.N, fit.K, fit.modelfit.resid, fit.modelfit.β, XX_inv, vcov)
end