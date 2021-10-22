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
    # Default standard errors homoscedastic
    function OLSModel(y::Vector, 
                      X::Matrix; 
                      vcov::vcov = vcovIID())
        n = size(X, 1)
        length(y) == n || error("y and x have differing numbers
        of observations.")
        Q, R = qr(X)
        new(y, X, vcov, Q, R)
    end
end


# Adding DataFrames functionality
function OLSModel(y::Vector, X::DataFrame; vcov::vcov = vcovIID())
    X = Matrix(X) # Backcompability warning requires Juila > 0.7
    return OLSModel(y, X, vcov = vcov)
end

mutable struct FitOutput
    β::Vector
    se_β::Vector
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
    se_β, σ_sq, vcov_matrix = se(resid, model, model.vcov)
    return FitOutput(β, se_β, resid, σ_sq, vcov_matrix)
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
        fit!(model))
end



