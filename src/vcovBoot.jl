struct vcovBoot <: vcov
    B::Int64
    cluster::Matrix
    n_cluster_vars::Int64
    function vcovBoot(B::Int64, cluster::Matrix)
        n_cluster_vars = size(cluster, 2)
        if n_cluster_vars > 2
            error("Only support one bootstrapped cluster.")
        end
            new(B, cluster, n_cluster_vars)
    end
end



function rademacher(N)
    D = rand(N) .> 0.5
    rademacher = D .* 1 .+ (1 .- D) .* -1 
    return rademacher
end

function rademacher(dims::Tuple)
   hcat(rademacher.(fill(dims[1], dims[2]))...) 
end

function inference(N::Int,
                   K::Int,
                   resid::Vector, 
                   β::Vector, 
                   XX_inv::Matrix,
                   X::Matrix,
                   vcov::vcovBoot)
    if vcov.n_cluster_vars != 1
        error("Only support one clustered variable Bootstrap atm")
    end
    B = vcov.B
    cluster_vals = vcov.cluster[:, 1]
    unique_clusters = unique(cluster_vals)
    n_clusters = length(unique_clusters)



    numeric_index_ids = Dict(unique_clusters .=> 1:n_clusters)
    cluster_ids = getindex.(Ref(numeric_index_ids), cluster_vals)
    
    radm_matrix = rademacher((N, B))
    
    # Perform inference for realised data
    se_r, pval_r, σ_sq_r, vcov_matrix_r = inference(
        N,
        K,
        resid,
        β,
        XX_inv,
        X,
        vcovCluster(vcov.cluster)
    )
    t_r = β ./ se_r
    B = vcov.B
    t_b = Vector{Vector{Float64}}(undef, B)
    se_b_vec = Vector{Vector{Float64}}(undef, B)
    for i in 1:B
        resid_b = resid .* radm_matrix[cluster_ids, i] 
        y_b = X * β .+ resid_b
        β_b = X \ y_b
        se_b, _, σ_sq_b, _ = inference(
            N,
            K,
            resid_b,
            β_b,
            XX_inv,
            X,
            vcovCluster(vcov.cluster)
        )
        t_b[i] = β_b ./ se_b
        se_b_vec[i] = se_b
    end
    pval_lower = (1/B) .* sum(t_b .< t_r, dims = 2)
    pval_upper = (1/B) .* sum(t_b .> t_r, dims = 2)
    pval = minimum.(pval_lower, pval_upper)

    return mean(se_b_vec, dims = 2), pval, σ_sq_r, vcov_matrix_r
end



# function getbootindices(vcov::vcovBoot)
#     cluster_vals = vcov.cluster[:, 1]
#     unique_cluster_vals = unique(vcov.cluster)
#     N_clusters = length(unique_cluster_vals)

#     new_clusters = sample(
#         unique_cluster_vals, 
#         length(unique_cluster_vals), 
#         replace = true)
#     boot_indices = Vector{Vector{Int64}}(undef, N_clusters)
#     for i in 1:N_clusters
#         boot_indices[i] = findall(x -> x == new_clusters[i], cluster_vals ) 
#     end
#     boot_indices = reduce(vcat, boot_indices)
#     return boot_indices
# end





# using StatsBase
# X = randn(20, 2)
# cl = sample(["a", "b", "c", "d", "e"], 20, replace = true)


# cluster_vals = cl
# unique_clusters = unique(cl)
# n_clusters = length(unique_clusters)
# cluster_indices = Vector{Vector{Int64}}(undef, n_clusters)

# contiguous_ids = Dict( unique(jj) .=> 1:length(unique(jj))  )
# jj .= getindex.(Ref(contiguous_ids),jj);


# index_ids = Dict(unique_clusters .=> 1:n_clusters)
# cluster_vals_2 = getindex.(Ref(index_ids), cluster_vals)

# cluster_vals
# unique_clusters
# findall(x -> x == unique_clusters[1], cluster_vals)
# for i in 1:n_clusters
#     cluster_indices[i] = findall(x -> x == unique_clusters[i], unique_clusters)
# end
#     cluster_indices = reduce(vcat, cluster_indices)
