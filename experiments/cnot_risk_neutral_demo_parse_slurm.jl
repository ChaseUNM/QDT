using LinearAlgebra, Printf, JLD2, Statistics
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures

###################################################################
# READ DATA FROM FILE
###################################################################

data_dir = "./data/cnot_risk_neutral_demo/"

opt_iters = 500
n_samples        = Int[]
infidelity_hist  = Vector{Float64}[]
runtimes         = Float64[]
final_infidelities = Float64[]


for filename in readdir(data_dir)

    full_path = joinpath(data_dir, filename)
    
    # Skip subdirectories
    if isdir(full_path)
        continue

    # Skip jld2 files
    elseif endswith(filename, ".jld2")
        continue

    # Skip files with the wrong number of iterations
    elseif !endswith(filename, "iters" * string(opt_iters) * ".out")
        continue
    end
    
    # Parse the # of samples from the filename
    pattern = r".+_samples(\d+)_.+"
    m = match(pattern, filename)
    push!(n_samples, parse(Int, m[1]))

    println(filename)

    # Parse convergence data from the file
    obj_vals = Float64[]
    open(full_path) do io

        risk_neutral_start = false
        data_start = false
        for line in eachline(io)

            # Skip forward to risk neutral control
            if !risk_neutral_start
                risk_neutral_start = contains(line, "RISK NEUTRAL CONTROL")
                continue
            end

            line_contains_iter = contains(line, "iter")
            
            # Skip forward until we see the first line with convergence data
            if !data_start
                data_start = line_contains_iter
                continue
            end

            # Skip likes containing the string "iter"
            if line_contains_iter
                continue
            end
            
            # Otherwise attempt to extract the objective value
            tokens = split(line)
            # Break once we hit an empty line
            if length(tokens) == 0
                break
            end
            val = 0;
            try
                val = parse(Float64, tokens[2])
            catch
                println(filename)
                println(line)
                val = parse(Float64, tokens[2])
            end
            push!(obj_vals, val)
        end

    end

    obj_vals = obj_vals ./ n_samples[end]
    push!(infidelity_hist, obj_vals)

    # Load the final infidelity of the controls 
    jld2_filename = filename[1:end-4] * ".jld2"
    I = load(joinpath(data_dir,jld2_filename), "infidelities")
    push!(final_infidelities, mean(I))

end


###################################################################
# PLOT CONVERGENCE
###################################################################

f = plot(
    title= "CNOT Risk Neutral (RN) Convergence", 
    titlefontsize=16, 
    xlabel="Iteration", xlabelfontsize=14,
    ylabel="Est. RN Infidelity", ylabelfontsize=14, 
    size=(600,500), 
    yscale=:log10,
    grid=true, gridlinewidth=2,
    tickfontsize=12,
    dpi=512, 
    left_margin = 8mm, 
    top_margin = 8mm, 
    bottom_margin = 8mm,
    right_margin = 8mm,
    linewidth=2,
    yticks=[1e-5,1e-4,1e-3,1e-2,1e-1,1]
)

for i = eachindex(n_samples)
    plot!(
        infidelity_hist[i],
        linewidth=2,
        label=@sprintf("%d samples", n_samples[i])
    )
end

# fig_dir = "./figures/cnot_risk_neutral_demo"
# filename_prefix = split(filename,".")[1]
# savefig(f, joinpath(fig_dir, filename_prefix * ".svg"))