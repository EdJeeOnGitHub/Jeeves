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
    tidy(fit::Jeeves.FittedOLSModel)
Inspired by R's broom::tidy. Returns a dataframe where each row is a variable
and its associated estimate.
"""
function tidy(fit::Jeeves.FittedOLSModel)
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
    tidy(model_list::Vector{FittedOLSModel}, model_name::Vector{String})
Method dispatch over a list of models and a vector of model identifiers
so we can just tidy over multiple broadcasted models.
"""
function tidy(model_list::Vector{FittedOLSModel}, model_names::Vector)
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



function TableCol(header, m::LinearModelFit;
                  stats=(:N=>Int∘nobs, "\$R^2\$"=>r2),
                  meta=(), kwargs...)

    # Initialize the column
    col  = RegCol(header)

    # Add the coefficients
    for (name, val, se, p) in zip(m.X_names, coef(m), m.modelfit.se_β, m.modelfit.pval)
        setcoef!(col, name, val, se)
        0.05 <  p <= .1  && star!(col[name], 1)
        0.01 <  p <= .05 && star!(col[name], 2)
                p <= .01 && star!(col[name], 3)
    end

    # Add in the fit statistics
    setstats!(col, OrderedDict(p.first=>p.second(m) for p in stats))

    # Add in the metadata
    setmeta!(col, OrderedDict(p.first=>p.second(m) for p in meta))

    return col

end
