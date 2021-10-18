abstract type vcov end

struct vcovIID <: vcov
end

struct vcovCluster <: vcov
    cluster::Vector
end

function findBcluster(cluster_var::Vector, X::Matrix, resid::Vector)
    unique_clusters = unique(cluster_var)

    n_clusters = length(unique(cluster_var))

    B_clusters = Vector{Matrix{Float64}}(undef, n_clusters)
    N, K = size(X, 1), size(X, 2)
    dof_adjustment = (n_clusters/(n_clusters - 1))*((N-1)/(N-K))
    for i in 1:n_clusters
        cluster_index = findall(cluster_var .== unique_clusters[i])
        X_cl = X[cluster_index, :]
        resid_cl = resid[cluster_index, :]
        B_cl = X_cl' * resid_cl * resid_cl' * X_cl
        B_clusters[i] = B_cl*dof_adjustment
    end
    B_hat = reduce(+, B_clusters)
    return B_hat
end

function se(fit::Fit, vcov::vcovCluster)
    cluster_var = vcov.cluster
    X, R = fit.X, fit.R
    resid = fit.modelfit.resid
    B_hat = findBcluster(cluster_var, X, resid)

    XX_inv = inv(cholesky(R' * R))
    vcov_matrix = XX_inv * B_hat * XX_inv
    se = sqrt.(diag(vcov_matrix))

    return se, fit.modelfit.σ_sq, vcov_matrix
end


function se(resid::Vector, fit::Model, vcov::vcovCluster)
    cluster_var = vcov.cluster
    X, R =  fit.X, fit.R

    B_hat = findBcluster(cluster_var, X, resid)

    XX_inv = inv(cholesky(R' * R))
    vcov_matrix = XX_inv * B_hat * XX_inv
    se = sqrt.(diag(vcov_matrix))
    σ_sq = sum(resid.^2)
    return se, σ_sq, vcov_matrix
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