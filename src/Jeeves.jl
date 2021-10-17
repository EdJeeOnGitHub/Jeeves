module Jeeves
    using LinearAlgebra



    export fit
    export fit!
    export coef
    export summary




    include("model.jl")
    include("vcov.jl")
    include("ols.jl")
end # module
