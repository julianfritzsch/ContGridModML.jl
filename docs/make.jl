using Documenter, ContGridModML

makedocs(sitename = "ContGridModML",
    modules = [ContGridModML],
    pages = [
        "Introduction" => "index.md",
        "API" => [
            "Public" => "api/public.md",
        ],
    ])

deploydocs(repo = "github.com/julianfritzsch/ContGridMod.jl.git")
