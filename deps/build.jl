using Pkg
using TOML
Pkg.add("PyPlot")
Pkg.add(url = "https://github.com/russelljjarvis/SpikeSynchrony.jl")
Pkg.add(url="https://github.com/paulmthompson/SpikeSorting.jl.git")
Pkg.add(url="https://github.com/paulmthompson/Spikes.jl.git")
Pkg.add(url="https://github.com/jkrumbiegel/ClearStacktrace.jl")
Pkg.add("Conda")
Pkg.add("PyCall")
run(`$(which pip) install -networkunit`)
valid_test_folder = joinpath(@__DIR__, "testfiles", "valid")

tml = TOML.tryparsefile("../Project.toml")
