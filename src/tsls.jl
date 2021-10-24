# I was in a rush okay because I rewrote OLS and SEs 3 times making 
# these beautiful hierarchies.

struct TSLSModel <: LinearModel
    y::Vector # outcome variable
    X::Matrix # Second stage variables (endog + exog)
    Z::Matrix # First stage variables (instrument + exog)
    Q_Z::Matrix
    R_Z::Matrix
    vcov::vcov
    X_names::Vector
    N::Int64
    K::Int64
    function TSLSModel(y::Vector, 
                       X_endog::Matrix, 
                       X_exog::Matrix, 
                       instruments::Matrix; 
                       vcov::vcov = vcovIID(),
                       X_names = nothing)
        N = length(y)
        Z = hcat(instruments, X_exog)
        X = hcat(X_endog, X_exog)
        Q_Z, R_Z = qr(Z)
        K = size(X, 2)
        if isnothing(X_names)
            X_names = vcat(
                "x_endog_" .* string.(1:size(X_endog, 2)),
                "x_exog_" .* string.(1:size(X_exog, 2))
            )
        end
        new(y, X, Z, Q_Z, R_Z, vcov, X_names, N, K) 
    end
end

# Adding DataFrames functionality
function TSLSModel(y::Vector, 
                    X_endog::DataFrame, 
                    X_exog::DataFrame, 
                    instruments::DataFrame; 
                    vcov::vcov = vcovIID())
    X_names = vcat(names(instruments), names(X_exog))
    X_endog = Matrix(X_endog)
    X_exog = Matrix(X_exog)
    instruments = Matrix(instruments)
    return TSLSModel(y, X_endog, X_exog, instruments, vcov = vcov, X_names = X_names)
end

struct FittedTSLSModel <: LinearModelFit
    y::Vector # outcome variable
    X::Matrix # Second stage variables (endog + exog)
    Z::Matrix # First stage variables (instrument + exog)
    Q::Matrix
    R::Matrix
    vcov::vcov
    X_names::Vector
    N::Int64
    K::Int64
    modelfit::FitOutput
    P_Z::Matrix
end


function fit!(model::TSLSModel)
    y = model.y
    X = model.X
    Z = model.Z
    R = model.R_Z
    ZZ_inv = inv(cholesky(R' * R))
    P_Z = Z * ZZ_inv * Z'
    β = inv(X' * P_Z * X) * X' * P_Z * y
    resid = y - X*β
    se_β, pval, σ_sq, vcov_matrix = inference(resid, 
                                              β,
                                              P_Z,
                                              model, model.vcov)
    return FitOutput(β, se_β, pval, resid, σ_sq, vcov_matrix), P_Z
end

function fit(model::TSLSModel)
    FittedTSLSModel(
        model.y,
        model.X,
        model.Z,
        model.Q_Z,
        model.R_Z,
        model.vcov,
        model.X_names,
        model.N,
        model.K,
        fit!(model)...
    )
end