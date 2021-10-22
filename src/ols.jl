"""
We distinguish between models objects and fitted model objects generally, 
for OLSModel specifically:


We separate OLSModel {LinearModel} from its fitted counterpart FittedOLSModel
 {LinearModelFit}  - OLSModel and FittedOLSModel are immutable structs. However,
 FittedOLSModel, whilst immutable, contains a mutable attribute 
 modelfit::FitOutput. fit!(model::OLSModel) returns an object of type FitOutput
 in place but fit(model::OLSModel) returns an object of type FittedOLSModel with
 updated modelfit::FitOutput.


 This means in simulations we can save memory allocation by just calling fit!
 without having to pass around y's, X's etc. It also means struct attributes 
 are immutable so we can't accidentally overwrite data whilst also leaving
 FitOutput accessible to manipulation after estimating models etc.
"""



"""
    OLSModel <: LinearModel

Object describing data and variance covariance matrix to be computed using an 
Ordinary Least Squares Model. Q, R are the QR decomposition pre-calculated and 
carried around by OLSModel.
"""
struct OLSModel <: LinearModel
    y::Vector
    X::Matrix
    vcov::vcov
    Q::Matrix # TODO: Create a lighweight version that doesn't carry data 
    R::Matrix
    X_names::Vector
    N::Int64
    K::Int64
    # Default standard errors homoscedastic
    function OLSModel(y::Vector, 
                      X::Matrix; 
                      vcov::vcov = vcovIID(),
                      X_names = nothing)
        N = size(X, 1)
        K = size(X, 2)
        length(y) == N || error("y and x have differing numbers
        of observations.")
        Q, R = qr(X)

        if isnothing(X_names)
            X_names = "x_" .* string.(1:size(X, 2))
        end
        new(y, X, vcov, Q, R, X_names, N, K)
    end
end


# Adding DataFrames functionality
function OLSModel(y::Vector, X::DataFrame; vcov::vcov = vcovIID())
    X_names = names(X)
    X = Matrix(X) # Backcompability warning requires Juila > 0.7
    return OLSModel(y, X, vcov = vcov, X_names = X_names)
end

mutable struct FitOutput
    β::Vector
    se_β::Vector
    pval::Vector
    resid::Vector
    σ_sq::Float64
    vcov_matrix::Matrix
end

struct FittedOLSModel <: LinearModelFit
    y::Vector
    X::Matrix
    vcov::vcov
    Q::Matrix
    R::Matrix
    X_names::Vector
    N::Int64
    K::Int64
    modelfit::FitOutput
end



"""
    fit!(model::OLSModel)

Use the QR decomposition to estimate y = X β and return FitOutput.
"""
function fit!(model::OLSModel)
    y = model.y
    X = model.X
    # QR decomposition for numerical stability
    Q = model.Q
    R = model.R
    # Find beta
    β = inv(R) * Q' * y 
    # SEs
    resid = y - X*β
    se_β, pval, σ_sq, vcov_matrix = inference(resid, β, model, model.vcov)
    return FitOutput(β, se_β, pval, resid, σ_sq, vcov_matrix)
end  




"""
    fit(model::OLSModel)

Instead of returning FitOutput, returns a FittedOLSModel.
"""
function fit(model::OLSModel)
    FittedOLSModel(
        model.y, 
        model.X, 
        model.vcov, 
        model.Q, 
        model.R,
        model.X_names,
        model.N,
        model.K,
        fit!(model))
end



