using LinearAlgebra, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures

###################################################################
# READ DATA FROM FILE
###################################################################

datapath = "./data/cnot_risk_neutral_demo/"
filename = "cnot_rn_5016777_samples60.out"
n_samples = parse(Int, split(filename,"samples")[end][1:end-4])

obj_vals = Float64[]

open(joinpath(datapath, filename)) do io

    risk_neutral_start = false
    data_start = false
    for line in eachline(io)

        # Skip forward to risk neutral control
        if !risk_neutral_start
            risk_neutral_start = contains(line, "RISK NEUTRAL CONTROL")
            continue
        end

        line_startswith_iter = startswith(line, "iter")
        
        # Skip forward until we see the first line with convergence data
        if !data_start
            data_start = line_startswith_iter
            continue
        end

        # Skip likes starting with iter
        if line_startswith_iter
            continue
        end
        
        # Otherwise attempt to extract the objective value
        println(line)
        tokens = split(line)
        # Break once we hit an empty line
        if length(tokens) == 0
            break
        end
        val = parse(Float64, tokens[2])
        push!(obj_vals, val)
    end

end

obj_vals = obj_vals ./ n_samples


###################################################################
# PLOT CONVERGENCE
###################################################################

title_str = @sprintf("CNOT Risk Neutral with %d Samples", n_samples)

f = plot(
    obj_vals,
    title= LaTeXString(title_str), 
    titlefontsize=16, 
    xlabel="Iteration", xlabelfontsize=14,
    ylabel="Sample Mean Infidelity", ylabelfontsize=14, 
    size=(600,500), 
    yscale=:log10,
    grid=true, gridlinewidth=2,
    tickfontsize=12,
    dpi=512, 
    left_margin = 8mm, 
    right_margin = 8mm,
    legend=false, linewidth=2
)

fig_dir = "./figures/cnot_risk_neutral_demo"
filename_prefix = split(filename,".")[1]
savefig(f, joinpath(fig_dir, filename_prefix * ".svg"))