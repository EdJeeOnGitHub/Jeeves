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



struct OLSModel <: LinearModel
    y::Vector
    X::Matrix
    vcov::vcov
    # Default standard errors homoscedastic
    function OLSModel(y::Vector, X::Matrix; vcov::vcov = vcov(:iid))
        n = size(X, 1)
        length(y) == n || error("y and x have differing numbers
        of observations.")
        new(y, X, vcov)
    end
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
    modelfit::FitOutput
end



"""
    fit!(model::OLSModel)

Use the QR decomposition to estimate y = X β and return FitOutput.
"""
function fit!(model::OLSModel)
    y = model.y
    X = model.X
    n = length(y)
    k = size(X, 2)
    # QR decomposition for numerical stability
    Q, R = qr(X)
    # Find beta
    β = inv(R) * Q' * y 
    # SEs
    resid = y - X * β
    σ_sq = sum(resid.^2) / (n - k)
    vcov_matrix = inv(cholesky(R' * R)) * σ_sq
    se_β = sqrt.(diag(vcov_matrix)) 

    return FitOutput(β, se_β, resid, σ_sq, vcov_matrix)
end  

"""
    fit(model::OLSModel)

Instead of returning FitOutput, returns a FittedOLSModel.
"""
function fit(model::OLSModel)
    FittedOLSModel(model.y, model.X, model.vcov, fit!(model))
end



