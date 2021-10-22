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

# @testset "Clustered SEs" begin
    
#     cluster_var_1 = repeat(["Kentucky", "Mass", "Illinois", "Florida"], 100) 
#     cluster_var_2 = repeat(["t_1", "t_2", "t_3", "t_4", "t_5"], 80)
#     cluster_var_3 = repeat(["village_1", "village_2"], 200)

#     using Random
#     cluster_matrix = Matrix(hcat(shuffle(cluster_var_1), shuffle(cluster_var_2), shuffle(cluster_var_3)))
#     cluster_matrix

#     X = randn(400, 5)
#     ϵ = randn(400, 1)
#     β = [1, 2, 3, 4, 5]
#     y = X * β    + ϵ

#     unique(cluster_matrix)

#     using Jeeves
#     test_clust = Jeeves.vcovCluster(cluster_matrix)
#     Rset = Jeeves.createRset(test_clust)

#     test_clust.cluster
#     function indicator_R(cluster_obj::vcovCluster, R)
#         R = Rset[2]
#         index_thin = findall(R .== 1)
#         unique(cluster_matrix[:, index_thin]) 

#     end



# end