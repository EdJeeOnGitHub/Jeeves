using Jeeves
using DataFrames:DataFrame
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

@testset "OLS perfect fit, default vcov" begin
    X = rand(1_000, 5)
    β = [ 1; 2; 3; 4; 5]
    y = X*β
    
    
    my_model = Jeeves.OLSModel(y, X)
    
    fitted_model = fit(my_model)
    model_coefs = coef(fitted_model)
    @test isapprox(model_coefs[1], β[1], rtol = rtol)
    @test isapprox(model_coefs[2], β[2], rtol = rtol)
    @test isapprox(model_coefs[3], β[3], rtol = rtol)
    @test isapprox(model_coefs[4], β[4], rtol = rtol)
    @test isapprox(model_coefs[5], β[5], rtol = rtol)
end

rtol = 0.0001
@testset "OLS perfect fit, DataFrame version" begin
    X = rand(1_000, 5)
    β = [ 1; 2; 3; 4; 5]
    y = X*β

    regression_df =  DataFrame(hcat(y, X), ["y", "x1", "x2", "x3", "x4", "x5"])
    my_model = Jeeves.OLSModel(regression_df[!, "y"], regression_df[!, ["x1", "x2", "x3", "x4", "x5"]], vcov = Jeeves.vcovIID())
    
    fitted_model = fit(my_model)
    model_coefs = coef(fitted_model)
    @test isapprox(model_coefs[1], β[1], rtol = rtol)
    @test isapprox(model_coefs[2], β[2], rtol = rtol)
    @test isapprox(model_coefs[3], β[3], rtol = rtol)
    @test isapprox(model_coefs[4], β[4], rtol = rtol)
    @test isapprox(model_coefs[5], β[5], rtol = rtol)
end

# Literally just testing if we get an answer at this stage lol...
@testset "Clustered SEs" begin
    N = 400   
    cluster_var_1 = repeat(["Kentucky", "Mass", "Illinois", "Florida"], Int(N/4)) 
    cluster_var_2 = repeat(["t_1", "t_2", "t_3", "t_4", "t_5"],Int(N/5))
    cluster_var_3 = repeat(1:50, Int(N/50))

    using Random
    cluster_matrix = Matrix(hcat(shuffle(cluster_var_1), 
                                 shuffle(cluster_var_2)))

    X = randn(N, 5)
    ϵ = randn(N, 1)
    β = [1, 2, 3, 4, 5]
    y = X * β    + ϵ

    model = Jeeves.OLSModel(y[:], X)
    model_fit = fit(model)
    SEs = Jeeves.inference(model_fit, Jeeves.vcovCluster(cluster_matrix))
    @test typeof(SEs[1]) == Vector{Float64}
end