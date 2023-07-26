using Documenter, ContGridModML

makedocs(sitename = "ContGridModML",
    authors = "Julian Fritzsch",
    modules = [ContGridModML],
    pages = [
        "Introduction" => "index.md",
        "Implementation" => [
            "Finite Element Method" => "implementation/finite.md",
            "General Implementation" => "implementation/general.md",
        ],
        "API" => [
            "Public" => "api/public.md",
            "Internal" => [
                "api/internal/dynamic.md",
                "api/internal/static.md",
                "api/internal/tools.md",
            ],
        ],
    ])

deploydocs(repo = "github.com/julianfritzsch/ContGridModML.jl.git", devbranch = "dev")
