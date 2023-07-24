using Documenter, ContGridModML

makedocs(sitename = "ContGridModML",
    authors = "Julian Fritzsch",
    modules = [ContGridModML],
    pages = [
        "Introduction" => "index.md",
        "API" => [
            "Public" => "api/public.md",
            "Internal" => [
                "api/internal/dynamic.md",
                "api/internal/tools.md",
            ],
        ],
    ])

deploydocs(repo = "github.com/julianfritzsch/ContGridModML.jl.git")
