
struct vcovCluster <: vcov
    cluster::Matrix
    n_cluster_vars::Int64
    function vcovCluster(cluster::Matrix)
        n_cluster_vars = size(cluster, 2)
        new(cluster, n_cluster_vars)
    end
end

# Do this
# http://cameron.econ.ucdavis.edu/research/CGM_twoway_ALL_13may2008.pdf

function findclusterid(cluster_var)
    cluster_id_dict = Dict(unique(cluster_var) .=> 1:length(unique(cluster_var)))
    cluster_id = getindex.(Ref(cluster_id_dict), cluster_var)
    return cluster_id
end



function createRset(cluster_obj::vcovCluster)
    n_cluster_vars = cluster_obj.n_cluster_vars
    ones_matrix = ones(n_cluster_vars, n_cluster_vars)
    triangular_matrix = LinearAlgebra.UpperTriangular(ones_matrix)
    # it's vectors all the way down
    perm_matrix = Vector{Vector{Vector{Int}}}(undef, n_cluster_vars)
    for i in 1:n_cluster_vars
        perm_matrix[i] = unique(permutations(triangular_matrix[i, :]))
    end
    # unnest vec of vec
    R_set = reduce(vcat, perm_matrix)
    return R_set
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


