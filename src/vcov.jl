abstract type vcov end

struct vcovIID <: vcov
end


function inference(resid::Vector, β::Vector, fit::Model, vcov::vcovIID)
    R = fit.R
    N = fit.N
    K = fit.K

    σ_sq = sum(resid.^2) / (N - K)
    vcov_matrix = inv(cholesky(R' * R)) * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    pval = 2 .* cdf.(TDist(fit.N - fit.K), -abs.(β ./ se))   
    return se, pval, σ_sq, vcov_matrix
end

# Method dispatch on fitted model.
function inference(fit::Fit, vcov::vcovIID)
    resid = fit.modelfit.resid
    β = fit.modelfit.β
    return return inference(resid, β, fit, vcov) 
end
