"""
    dummy_matrix(df::DataFrame, column)


Takes a DataFrame and column name and expands factors
into dummy columns - helps to convert columns to "colname_colvalue" since values
are used as column names in the dummy DataFrame.
"""
function dummy_matrix(df::DataFrame, column)
    colnames = permutedims(df[!, column])
    dummy_m = unique(df[!, column]) .== colnames
    dummy_df = DataFrame(Matrix(permutedims(dummy_m)), convert(Vector{String}, string.(unique(colnames[:]))))
    return dummy_df
end

"""
    tidy(fit::LinearModelFit)
Inspired by R's broom::tidy. Returns a dataframe where each row is a variable
and its associated estimate.
"""
function tidy(fit::T) where {T <: Fit}
    X_names = fit.X_names
    β = fit.modelfit.β
    se_β = fit.modelfit.se_β
    pval = fit.modelfit.pval
    df = DataFrame(
        term = X_names,
        β = β,
        se_β = se_β,
        t_stat = β ./ se_β,
        p_val = pval    
        )
    return df
end

"""
    tidy(model_list::Vector{LinearModelFit}, model_name::Vector{String})
Method dispatch over a list of models and a vector of model identifiers
so we can just tidy over multiple broadcasted models.
"""
function tidy(model_list::Vector{T} where {T <: Fit}, model_names::Vector) 
    length(model_list) == length(model_names) || error("Model - Name Dimension Mismatch")
    tidy_model_list = Vector{DataFrame}(undef, length(model_list)) 
    for i in 1:length(model_list)
        df = tidy(model_list[i])
        df[!, "model"] .= model_names[i]
        tidy_model_list[i] =  df
    end
    tidy_df = reduce(vcat, tidy_model_list)
    return tidy_df
end
"""
    tidy(model_list::Vector{T} where {T <: Fit}) 
Tidy a list of models and generate model_i names automatically.
"""
function tidy(model_list::Vector{T} where {T <: Fit}) 
    tidy_model_list = Vector{DataFrame}(undef, length(model_list)) 
    for i in 1:length(model_list)
        df = tidy(model_list[i])
        df[!, "model"] .= "model_" * string(i)
        tidy_model_list[i] =  df
    end
    tidy_df = reduce(vcat, tidy_model_list)
    return tidy_df
end



function TableCol(header, 
                  m::LinearModelFit;
                  drop = [""],
                  keep = [""],
                  stats=(:N=>Int∘nobs, "\$R^2\$"=>r2),
                  meta=())
    # Initialize the column
    col  = RegCol(header)
    keep_used = !(keep == [""])
    # if we use keep, add everything to drop and remove elements in keep 
    if keep_used
        drop = deepcopy(m.X_names)
        deleteat!(drop,  findall(drop .== keep))
    end
    # Add the coefficients
    for (name, val, se, p) in zip(m.X_names, coef(m), m.modelfit.se_β, m.modelfit.pval)
        if !(name in drop)
            setcoef!(col, name, val, se)
            0.05 <  p <= .1  && star!(col[name], 1)
            0.01 <  p <= .05 && star!(col[name], 2)
                    p <= .01 && star!(col[name], 3)
        end
    end

    # Add in the fit statistics
    setstats!(col, OrderedDict(p.first=>p.second(m) for p in stats))

    # Add in the metadata
    setmeta!(col, OrderedDict(p.first=>p.second(m) for p in meta))

    return col

end
