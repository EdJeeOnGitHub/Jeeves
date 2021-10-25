using Revise
using Jeeves
using DataFrames:DataFrame
using Test
using RDatasets

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
    inference = Jeeves.inference(fitted_model, vcovIID())
    SEs = inference[1]
    model_coefs = coef(fitted_model)
    @test isapprox(model_coefs[1], β[1], rtol = rtol)
    @test isapprox(model_coefs[2], β[2], rtol = rtol)
    @test isapprox(model_coefs[3], β[3], rtol = rtol)
    @test isapprox(model_coefs[4], β[4], rtol = rtol)
    @test isapprox(model_coefs[5], β[5], rtol = rtol)

    @test isless(SEs[1], 1e-10)
    @test isless(SEs[2], 1e-10)
    @test isless(SEs[3], 1e-10)
    @test isless(SEs[4], 1e-10)
    @test isless(SEs[5], 1e-10)
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

@testset "TSLS and Anderson Rubin works" begin
    N = 10_000
    confounder = randn(N)
    X_exog = randn(N, 3)
    X_beta = [1; 2; 3]
    β_endog = [0; 0]
    Z = randn(N, 2)
    X_endog = randn(N, 2) .+ confounder .+ Z
    Y = X_exog * X_beta + X_endog*β_endog + confounder + randn(N, 1)
    IV_model = Jeeves.TSLSModel(Y[:], X_endog, X_exog, Z)
    IV_fit = fit(IV_model)
    separate_inference = Jeeves.inference(IV_fit, vcovIID())
    model_coefs = coef(IV_fit)
    @test isapprox(model_coefs[1], 1, rtol = 1e-1)
    @test isapprox(model_coefs[2], -1, rtol = 1e-1)
    @test isapprox(model_coefs[3], 1, rtol = 1e-1)
    @test isapprox(model_coefs[4], 2, rtol = 1e-1)
    @test isapprox(model_coefs[5], 3, rtol = 1e-1)


    # Testing AndersonRubinCI returns results
    β_grid = Iterators.product(-1.:0.5:0., -1.:0.5:0.)
    AR_test = Jeeves.AndersonRubinCI(β_grid, IV_fit)
    @test size(AR_test, 1) == 9
    @test size(AR_test[1], 1) == 5
end

@testset "JIVE works" begin
    N = 10_000
    confounder = randn(N)
    X_exog = randn(N, 3)
    X_beta = [1; 2; 3]
    β_endog = [1; -1]
    Z = randn(N, 2)
    X_endog = randn(N, 2) .+ confounder .+ Z
    Y = X_exog * X_beta + X_endog*β_endog + confounder + randn(N, 1)
    IV_model = Jeeves.TSLSModel(Y[:], X_endog, X_exog, Z)
    JIVE_fit = Jeeves.jive(IV_model)

    JIVE_fit.modelfit.se_β

    separate_inference = Jeeves.inference(JIVE_fit, vcovIID())
    @test isequal(JIVE_fit.modelfit.se_β, separate_inference[1])
    model_coefs = coef(JIVE_fit)
    @test isapprox(model_coefs[1], 1, rtol = 1e-1)
    @test isapprox(model_coefs[2], -1, rtol = 1e-1)
    @test isapprox(model_coefs[3], 1, rtol = 1e-1)
    @test isapprox(model_coefs[4], 2, rtol = 1e-1)
    @test isapprox(model_coefs[5], 3, rtol = 1e-1)
end

# Kinda hard to test these functions.
@testset "tidying" begin
    X = rand(1_000, 5)
    β = [ 1; 2; 3; 4; 5]
    y = X*β
    
    
    my_model = broadcast(x -> Jeeves.OLSModel(y, X), 1:5)
    fitted_models = fit.(my_model)
    single_tidy = tidy(fitted_models[1])
    many_tidy = tidy(fitted_models, collect(1:5))


    tex_table = Jeeves.TableCol("ed", fitted_models[1], drop = ["x_1"])
    @test length(tex_table.data) == 6
    @test typeof(single_tidy) == DataFrame
    @test typeof(many_tidy) == DataFrame
end



@testset "F-test matches t-stat squared" begin
    X = hcat(repeat([1.0], 1_000), rand(1_000, 1))
    β = [1, 5]
    y =  X*β + randn(1_000)
    model = Jeeves.OLSModel(y, X)    
    ols_fit = fit(model)
    tidy_fit = tidy(ols_fit)
    F_stat, _, _ = Jeeves.Ftest(ols_fit)
    t_stat_sq = tidy_fit[2, "t_stat"]^2

    @test isapprox(F_stat, t_stat_sq)
end    

# Literally just testing if we get an answer at this stage lol...
@testset "Clustered SEs Run" begin
    N = 4000   
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



@testset "Clustered SEs Match Sandwich" begin
    PublicSchools = dropmissing(dataset("sandwich", "PublicSchools"))
    PublicSchools[!, "fake_cluster"] .= repeat(1:5, 10)
    PublicSchools[!, "const"] .= 1.0
    PS_model = Jeeves.OLSModel(PublicSchools[!, "Expenditure"],
                               PublicSchools[!, ["const", "Income"]],
                               vcov = Jeeves.vcovCluster(
                                   Matrix(PublicSchools[!, ["State",
                                                            "fake_cluster"]])))
    PS_fit = fit(PS_model)
    PS_coef = coef(PS_fit)
    PS_SE = PS_fit.modelfit.se_β
    # Quick coef checks vs R package
    @test isapprox(-151.265090, PS_coef[1], rtol = rtol)
    @test isapprox(0.068939, PS_coef[2], rtol = rtol)

    # SEs
    # R's sandwich defaults to HC1 169.594852, 0.023020 
    @test isapprox(169.594852, PS_SE[1], rtol = rtol)
    @test isapprox(0.023020, PS_SE[2], rtol = rtol)
end