module Jeeves
    using LinearAlgebra, Combinatorics
    using Optim, Distributions


    export fit
    export fit!
    export coef
    export summary




    include("model.jl")
    include("vcov.jl")
    include("vcovCluster.jl")
    include("ols.jl")
    include("tsls.jl")
    include("probit.jl")

end # module
