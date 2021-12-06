struct vcovHetero <: vcov
    ssc::String
    function vcovHetero(; ssc::String = "HC1")
        new(ssc) 
    end
end



function inference(N::Int,
                   K::Int,
                   resid::Vector,
                   β::Vector,
                   XX_inv::Matrix,
                   X::Matrix,
                   vcov::vcovHetero)

    resid_sq = resid.^2
    vcov_matrix = ((N-1)/(N-K)) * XX_inv * (X' * diagm(resid_sq) * X) * XX_inv

    se = sqrt.(diag(vcov_matrix))
    pval = 2 .* cdf.(TDist(N - K), -abs.(β ./ se))   
    σ_sq = sum(resid.^2) / (N - K)
    return se, pval, σ_sq, vcov_matrix
end
