# I was in a rush okay because I rewrote OLS and SEs 3 times making 
# these beautiful hierarchies.

struct TSLSModel <: LinearModel
    y::Vector
    X_endog::Matrix
    X_exog::Matrix
    Z::Matrix
    Q_Z::Matrix
    R_Z::Matrix
    vcov::vcov
    function TSLSModel(y::Vector, X_endog::Matrix, X_exog::Matrix, Z::Matrix; vcov::vcov = vcovIID)
       Q_Z, R_Z = qr(Z)
       new(y, X_endog, X_exog, Z, vcov, Q_Z, R_Z) 
    end
end



struct FittedTSLSModel <: LinearModelFit
    y::Vector
    X_endog::Matrix
    X_exog::Matrix
    Z::Matrix
    Q_Z::Matrix
    R_Z::Matrix
    vcov::vcov
    modelfit::FitOutput
end


function fit!(model::TSLSModel)
    y = model.y
    X_endog = model.X_endog
    X_exog = model.X_exog
    Z = model.Z
    R = model.R_Z
    ZZ_inv = inv(cholesky(R' * R))
    P_Z = Z * ZZ_inv * Z'
    β = inv(X' * P_Z * X) * X' * P_Z * y
    # residuals and controls is tomorrow's problem
    # TODO
    return FitOutput(β, [0], [0], [0.0], [0 0; 0 0])
end

function fit(model::TSLSModel)
    FittedTSLSModel(
        model.y,
        model.X_endog,
        model.X_exog,
        model.Z,
        model.Q_Z,
        model.R_Z,
        model.vcov,
        fit!(model)
    )
end