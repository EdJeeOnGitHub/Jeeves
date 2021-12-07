struct vcovBoot <: vcov
    B::Int64
    cluster::Matrix
    n_cluster_vars::Int64
    β_0::Float64
    function vcovBoot(B::Int64, cluster::Matrix; β_0 = 0.0)
        n_cluster_vars = size(cluster, 2)
        if n_cluster_vars > 2
            error("Only support one bootstrapped cluster.")
        end
        new(B, cluster, n_cluster_vars, β_0)
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
    β_0 = vcov.β_0
    # if length(β_0) != K && length(β_0) == 1
    #     β_0 = fill(β_0[1], K)
    # end
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
    t_r = (β .- β_0) ./ se_r
    B = vcov.B
    t_b = Matrix{Float64}(undef, (B, K))
    se_b_vec = similar(t_b)
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
        t_b[i, :] = (β_b .- β) ./ se_b
        se_b_vec[i, :] = se_b
    end
    pval_lower = (1/B) .* sum(t_b' .> t_r, dims = 2)
    pval_upper = (1/B) .* sum(t_b' .< t_r, dims = 2)
    pval = 2 .*  minimum(hcat(pval_lower, pval_upper), dims = 2)
    return mean(se_b_vec, dims = 1)[:], pval[:], σ_sq_r, vcov_matrix_r
end


