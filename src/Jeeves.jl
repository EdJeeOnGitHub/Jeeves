module Jeeves
    using LinearAlgebra, Combinatorics
    using Optim, Distributions
    using DataFrames: DataFrame
    using TexTables, OrderedCollections
    using RDatasets


    export fit
    export fit!
    export coef
    export summary
    export tidy

    export vcovIID, vcovCluster

    ## Types ##
    """
    `F <: ModelFit` specifies whether the model describes a target 
    estimand or a fitted model with sampling variance describing an 
    estimator.
    """
    abstract type ModelFit end
    struct Estimator <: ModelFit end
    struct Estimand <: ModelFit end


    abstract type ModelType end
    abstract type LinearModel <: ModelType  end
    abstract type GeneralisedLinearModel <: ModelType end

    abstract type Model{M<:ModelType, F<:ModelFit} end




    abstract type vcov end

    ## Modules ##
    include("model.jl")
    include("ols.jl")
    include("tsls.jl")
    include("probit.jl")
    include("vcov.jl")
    include("vcovCluster.jl")
    include("utils.jl")
end # module
