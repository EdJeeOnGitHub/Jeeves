
struct vcovCluster <: vcov
    cluster::Matrix
    n_cluster_vars::Int64
    ssc::String
    # defaults to HC1 for now
    function vcovCluster(cluster::Matrix)
        n_cluster_vars = size(cluster, 2)
        ssc = "HC1"
        new(cluster, n_cluster_vars, ssc)
    end
end


"""
    findsinglecluster(cluster_var::Vector, X::Matrix, resid::Vector)
If there's a single cluster variable this is a much quicker implementation.
"""
function findsinglecluster(cluster_var::Vector, X::Matrix, resid::Vector)
    unique_clusters = unique(cluster_var)
    n_clusters = length(unique(cluster_var))
    B_clusters = Vector{Matrix{Float64}}(undef, n_clusters)
    N, K = size(X, 1), size(X, 2)
    dof_adjustment = (n_clusters/(n_clusters - 1))*((N-1)/(N-K))
    for i in 1:n_clusters
        cluster_index = findall(cluster_var .== unique_clusters[i])
        X_cl = X[cluster_index, :]
        resid_cl = resid[cluster_index, :]
        B_cl = (X_cl' * resid_cl) * (resid_cl' * X_cl)
        B_clusters[i] = B_cl*dof_adjustment
    end
    B_hat = reduce(+, B_clusters)
    return B_hat
end


I_R(R) = findall(R .== 1)
"""
    createRset(cluster_obj::vcovCluster)

Creates a set of D-length vectors where D corresponds to clustering dimension 
size. R = {r: r_d in {0, 1}, d = 1, 2,..., D} (page 9 Cameron, Gelbach, Miller)

These are the permutations of all the clustering dimensions we want to compare
across.
"""
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


"""
    createBtilde(cluster_matrix, indicators)
Creates a clustered B_tilde matrix as described in Cameron, Gelbach,
and Miller (2008)
(http://cameron.econ.ucdavis.edu/research/CGM_twoway_ALL_13may2008.pdf)


For now small sample correction defaults to HC1 (which is naughty).
"""
function createBtilde(X::Matrix, 
                      resid::Vector, 
                      N::Int64, 
                      cluster_matrix, 
                      indicators)
    K = size(X, 2)
    # Initialise B_r for each r in the datset.
    B_R = Vector{Matrix}(undef, length(indicators))
    for I in 1:length(indicators)
        # Only keep cluster dimensions we care about
        subset_cluster_matrix = cluster_matrix[:, indicators[I]]
        # Find each unique cluster tuple/pair in the dataset 
        unique_clusters = unique(subset_cluster_matrix, dims = 1) 
        cluster_G_size = size(unique_clusters, 1)
        B_G = Vector{Matrix}(undef, cluster_G_size)
        for i in 1:size(unique_clusters, 1)
            # iterate through each unique cluster tuple and select only Xs and
            # resids corresponding to that cluster tuple. all(cond, dims = 2) 
            # means we check that each ROW of the matrix meets the condition.
            cluster_selector = all(subset_cluster_matrix .== unique_clusters[i:i, :], dims = 2)[:]
            X_g = X[cluster_selector, :]
            resid_g = resid[cluster_selector]
            # Create B_G(I_r, ij) using these Xs and resids
            B_G[i] = (X_g' * resid_g) * (resid_g' * X_g) 
        end
        # Create odd even inclusion adjustment
        D = length(indicators[I])
        small_sample_correction = (cluster_G_size/(cluster_G_size - 1))*((N - 1)/(N-K))
        sum_adjustment = (-1)^(D+1)*small_sample_correction
        B_R[I] = sum_adjustment*reduce(+, B_G)
    end
    B_tilde = sum(B_R)
    return B_tilde
end

"""
    mat_posdef_fix(X::Matrix; tol = 1e-10)
Make matrix posdef if we have negative variances on the diagonal.

Proposed by Cameron, Gelbach, Miller (2011).
"""
function mat_posdef_fix(X::Matrix; tol = 1e-10)
    if any(diag(X) .< tol)
        e_vals, e_vecs = eigen(Symmetric(X))
        e_vals[e_vals .<= tol] .= tol
        X = e_vecs * Diagonal(e_vals) * e_vecs'
    end
    return X
end

"""
Clustered Standard errors inference.
"""
function inference(N::Int,
                   K::Int,
                   resid::Vector, 
                   β::Vector, 
                   XX_inv::Matrix,
                   X::Matrix,
                   vcov::vcovCluster)
    cluster_matrix = vcov.cluster

    n_cluster_vars = size(cluster_matrix, 2)
    if n_cluster_vars == 1
        B_tilde = findsinglecluster(cluster_matrix[:], X, resid)
    else 
        Rset = createRset(vcov)
        indicators = I_R.(Rset)
        B_tilde = createBtilde(X, resid, N, cluster_matrix, indicators)
    end

    vcov_matrix = XX_inv * B_tilde * XX_inv
    vcov_matrix = mat_posdef_fix(vcov_matrix)
    se = sqrt.(diag(vcov_matrix))

    pval = 2 .* cdf.(TDist(N - K), -abs.(β ./ se))   
    σ_sq = sum(resid.^2) / (N - K)
    return se, pval, σ_sq, vcov_matrix
end

