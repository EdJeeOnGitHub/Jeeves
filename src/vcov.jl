abstract type vcov end

struct vcovIID <: vcov
end


function inference(fit::Fit, vcov::vcovIID)
    R = fit.R
    σ_sq = fit.modelfit.σ_sq
    vcov_matrix = inv(cholesky(R' * R)) * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    pval = 2 .* cdf.(TDist(fit.N - fit.K), -abs.(fit.modelfit.β ./ se))   
    return se, pval, σ_sq, vcov_matrix
end

function inference(resid::Vector, β::vector, fit::Model, vcov::vcovIID)
    R = fit.R
    N = fit.N
    K = fit.K

    σ_sq = sum(resid.^2) / (N - K)
    vcov_matrix = inv(cholesky(R' * R)) * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    pval = 2 .* cdf.(TDist(fit.N - fit.K), -abs.(β ./ se))   
    return se, pval, σ_sq, vcov_matrix
end
