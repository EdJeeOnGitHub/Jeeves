module Jeeves
    using LinearAlgebra, Combinatorics



    export fit
    export fit!
    export coef
    export summary




    include("model.jl")
    include("vcov.jl")
    include("vcovCluster.jl")
    include("ols.jl")
end # module
