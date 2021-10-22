module Jeeves
    using LinearAlgebra, Combinatorics
    using Optim, Distributions
    using DataFrames: DataFrame
    using TexTables, OrderedCollections

    export fit
    export fit!
    export coef
    export summary
    export tidy

    export vcovIID, vcovCluster

    include("model.jl")
    include("vcov.jl")
    include("vcovCluster.jl")
    include("ols.jl")
    include("tsls.jl")
    include("probit.jl")
    include("utils.jl")
end # module
