using Jeeves
using Test


rtol = 0.0001
@testset "OLS perfect fit" begin
    X = rand(1_000, 5)
    β = [ 1; 2; 3; 4; 5]
    y = X*β
    
    
    my_model = Jeeves.OLSModel(y, X, vcov = Jeeves.vcovIID())
    
    fitted_model = fit(my_model)
    model_coefs = coef(fitted_model)
    @test isapprox(model_coefs[1], β[1], rtol = rtol)
    @test isapprox(model_coefs[2], β[2], rtol = rtol)
    @test isapprox(model_coefs[3], β[3], rtol = rtol)
    @test isapprox(model_coefs[4], β[4], rtol = rtol)
    @test isapprox(model_coefs[5], β[5], rtol = rtol)
end


@testset "Clustered SEs" begin
    
    cluster_var_1 = repeat(["Kentucky", "Mass", "Illinois", "Florida"], 100) 
    cluster_var_2 = repeat(["t_1", "t_2", "t_3", "t_4", "t_5"], 80)
    cluster_var_3 = repeat(["village_1", "village_2"], 200)

    using Random
    cluster_matrix = Matrix(hcat(shuffle(cluster_var_1), shuffle(cluster_var_2), shuffle(cluster_var_3)))
    cluster_matrix

    X = randn(400, 5)
    ϵ = randn(400, 1)
    β = [1, 2, 3, 4, 5]
    y = X * β    + ϵ

    unique(cluster_matrix)

    using Jeeves
    test_clust = Jeeves.vcovCluster(cluster_matrix)
    createRset(test_clust)
end