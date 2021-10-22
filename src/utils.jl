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