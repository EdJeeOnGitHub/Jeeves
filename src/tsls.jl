
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
    K_endog::Int64
    K_exog::Int64
    K_instrument::Int64
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
        K_endog = size(X_endog, 2)
        K_exog = size(X_exog, 2)
        K_instrument = size(instruments, 2)
        if isnothing(X_names)
            X_names = vcat(
                "x_endog_" .* string.(1:size(X_endog, 2)),
                "x_exog_" .* string.(1:size(X_exog, 2))
            )
        end
        new(y, X, Z, Q_Z, R_Z, vcov, X_names, N, K, K_endog, K_exog, K_instrument) 
    end
end

# Adding DataFrames functionality
function TSLSModel(y::Vector, 
                    X_endog::DataFrame, 
                    X_exog::DataFrame, 
                    instruments::DataFrame; 
                    vcov::vcov = vcovIID())
    X_names = vcat(names(X_endog), names(X_exog))
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
    K_endog::Int64
    K_exog::Int64
    K_instrument::Int64
    modelfit::FitOutput
    P_Z::Matrix
end

struct FittedJIVEModel <:LinearModelFit
    y::Vector # outcome variable
    X::Matrix # Second stage variables (endog + exog)
    Z::Matrix # First stage variables (instrument + exog)
    Q::Matrix
    R::Matrix
    vcov::vcov
    X_names::Vector
    N::Int64
    K::Int64
    K_endog::Int64
    K_exog::Int64
    K_instrument::Int64
    modelfit::FitOutput
    P_Z::Matrix
    X_jive::Matrix
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
        model.K_endog,
        model.K_exog,
        model.K_instrument,
        fit!(model)...
    )
end



function AndersonRubinCI!(β_AR, 
                          fit_obj::Jeeves.FittedTSLSModel;
                          additional_controls = nothing,
                          signif_level = 0.05)
    K_instrument = fit_obj.K_instrument
    K_endog = fit_obj.K_endog
    X_endog = fit_obj.X[:, 1:K_endog]
    X_exog_only = fit_obj.Z[:, (K_instrument + 1):end]
    X_exog = fit_obj.Z
    AR_resid = fit_obj.y - X_endog*β_AR  
    if !isnothing(additional_controls)
        X_exog = hcat(fit_obj.Z, additional_controls)
        X_exog_only = hcat(X_exog_only, additional_controls)
    end
    AR_model = OLSModel(AR_resid, X_exog, vcov = fit_obj.vcov)
    restricted_AR_model = OLSModel(AR_resid, X_exog_only, vcov = fit_obj.vcov)
    AR_fit = fit(AR_model)
    restricted_AR_fit = fit(restricted_AR_model) 
    AR_F_vals = Ftest(AR_fit, restricted_AR_fit, signif_level = signif_level)
    return [β_AR..., AR_F_vals...]
end



function AndersonRubinCI(β_grid, fit_obj::FittedTSLSModel; additional_controls = nothing)
    β_grid = collect(β_grid)
    K_endog = fit_obj.K_endog
    K_β =  length(β_grid[1])
    K_β == K_endog || error("Grid dimension doesn't match K_endog")
    AR_results = Vector(undef, length(β_grid)) 
    for i in 1:length(β_grid)
        AR_results[i] = AndersonRubinCI!(
            [β_grid[i]...], 
            fit_obj, 
            additional_controls = additional_controls)
    end
    return AR_results
end


function jive!(model::TSLSModel)
    y = model.y
    X = model.X
    Z = model.Z
    R = model.R_Z
    ZZ_inv = inv(cholesky(R' * R))
    P_Z = Z * ZZ_inv * Z'
    h_i = diag(P_Z)
    π = ZZ_inv * (Z' * X)
    X_jive = (Z * π - h_i .* X)./(1 .- h_i)
    XX_inv = inv(X_jive' * X)
    β = XX_inv * (X_jive' * y)
    resid = y - X*β

    se_β, pval, σ_sq, vcov_matrix = inference(model.N, 
                                              model.K, 
                                              resid, 
                                              β, 
                                              XX_inv, 
                                              X_jive, 
                                              model.vcov)
    return FitOutput(β, se_β, pval, resid, σ_sq, vcov_matrix), P_Z, X_jive
end


function jive(model::TSLSModel)
    FittedJIVEModel(
        model.y,
        model.X,
        model.Z,
        model.Q_Z,
        model.R_Z,
        model.vcov,
        model.X_names,
        model.N,
        model.K,
        model.K_endog,
        model.K_exog,
        model.K_instrument,
        jive!(model)...
    )
end
