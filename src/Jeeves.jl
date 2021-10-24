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
    abstract type Model end
    abstract type LinearModel <: Model  end
    abstract type GeneralisedLinearModel <: Model end


    abstract type Fit <: Model end
    abstract type LinearModelFit <: Fit end


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
