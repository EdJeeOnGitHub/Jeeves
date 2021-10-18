abstract type vcov end

struct vcovIID <: vcov
end


function se(fit::Fit, vcov::vcovIID)
    R = fit.R
    σ_sq = fit.modelfit.σ_sq
    vcov_matrix = inv(cholesky(R' * R)) * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    return se, σ_sq, vcov_matrix
end

function se(resid::Vector, fit::Model, vcov::vcovIID)
    y, X, R = fit.y, fit.X, fit.R
    n = length(y)
    k = size(X, 2)

    σ_sq = sum(resid.^2) / (n - k)
    vcov_matrix = inv(cholesky(R' * R)) * σ_sq
    se = sqrt.(diag(vcov_matrix)) 
    return se, σ_sq, vcov_matrix
end