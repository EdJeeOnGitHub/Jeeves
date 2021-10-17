using Documenter, Jeeves

makedocs(
    modules = [Jeeves],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "=",
    sitename = "Jeeves.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/EdJeeOnGitHub/Jeeves.jl.git",
    push_preview = true
)
