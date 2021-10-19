

struct ProbitModel <: GeneralisedLinearModel
    y::Vector
    X::Matrix
end

struct FittedProbitModel <: Fit
    y::Vector
    X::Matrix
    modelfit::FitOutput
end



function fit!(model::ProbitModel)
    X = model.X
    
    init_β = repeat([0.0], size(X, 2))

    f(β) = loglikelihood(model, β)
    mle_result = optimize(f, init_β, LBFGS())
    β = mle_result.minimizer
    # forgive me father for I have sinned
    # TODO
    return FitOutput(β, [0], [0], [0.0], [0 0; 0 0])
end


function fit(model::ProbitModel)
    FittedProbitModel(
        model.y, 
        model.X, 
        fit!(model))
end

function loglikelihood(model::ProbitModel, β)
    μ = model.X*β
    y = model.y
    ϕ = Normal()
    ll = sum(y .* log.(cdf.(ϕ, μ)) .+ (1 .- y) .* log.(1 .- cdf.(ϕ, μ)))
    return -ll
end


