
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

cluster_var_1 = repeat(["Kentucky", "Mass", "Illinois", "Florida"], 100) 
cluster_var_2 = repeat(["t_1", "t_2", "t_3", "t_4", "t_5"], 80)
cluster_var_3 = repeat(["village_1", "village_2"], 200)

using Random
cluster_matrix = Matrix(hcat(shuffle(cluster_var_1), shuffle(cluster_var_2), shuffle(cluster_var_3)))
cluster_matrix

X = randn(400, 5)
ϵ = randn(400, 1)
β = [1, 2, 3, 4, 5]
y = X * β    + ϵ

unique(cluster_matrix)


function findclusterid(cluster_var)
    cluster_id_dict = Dict(unique(cluster_var) .=> 1:length(unique(cluster_var)))
    cluster_id = getindex.(Ref(cluster_id_dict), cluster_var)
    return cluster_id
end



ones(, 5)

unique(permutations([1, 1, 1]))
unique(permutations([0, 1, 1]))
unique(permutations([0, 0, 1]))


for a in 1:size_R
    println(a)
end

using LinearAlgebra

function createRset(cluster_obj::vcovCluster)
    n_cluster_vars = cluster_obj.n_cluster_vars
    n_cluster_vars = 3
    size_R = 2^n_cluster_vars - 1
    ones_matrix = ones(n_cluster_vars, n_cluster_vars)
    triangular_matrix = LinearAlgebra.UpperTriangular(ones_matrix)
    # it's vectors all the way down
    perm_matrix = Vector{Vector{Vector{Int}}}(undef, n_cluster_vars)
    R_set = Vector{Vector{Float64}}(undef, size_R)
    for i in 1:n_cluster_vars
        perm_matrix[i] = unique(permutations(triangular_matrix[i, :]))
    end
    perm_matrix

    # unnest vec of vec

    getindex.(perm_matrix, 1)
    reduce(, perm_matrix)
    collect.(unique(permutations.(triangular_matrix)))
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


