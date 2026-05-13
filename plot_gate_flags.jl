using Plots
#!/usr/bin/env julia
# Quick utility to plot a mapping from numeric x -> Dict{Gate,0/1}.
# Writes two PNGs to the current directory: gate_flags_lines.png and gate_flags_heatmap.png

# Data taken from the user's message
# pairs = [
#     (0.06812920690579612, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
#     (0.046415888336127795, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
#     (0.03162277660168379,  Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
#     (0.021544346900318846, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
#     (0.01467799267622069,  Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>0, "PauliX"=>1)),
#     (0.010000000000000002, Dict("PauliZ"=>0, "PauliY"=>1, "Hadamard"=>0, "PauliX"=>1)),
#     (0.006812920690579608, Dict("PauliZ"=>1, "PauliY"=>0, "Hadamard"=>0, "PauliX"=>0)),
#     (0.004641588833612782, Dict("PauliZ"=>0, "PauliY"=>0, "Hadamard"=>0, "PauliX"=>0)),
# ]

pairs = [
    (0.1, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
    (0.06812920690579612, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
    (0.046415888336127795, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
    (0.03162277660168379,  Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>1, "PauliX"=>1)),
    (0.021544346900318846, Dict("PauliZ"=>1, "PauliY"=>1, "Hadamard"=>0, "PauliX"=>1)),
    (0.01467799267622069,  Dict("PauliZ"=>0, "PauliY"=>1, "Hadamard"=>0, "PauliX"=>1)),
    (0.010000000000000002, Dict("PauliZ"=>1, "PauliY"=>0, "Hadamard"=>0, "PauliX"=>0)),
    (0.006812920690579608, Dict("PauliZ"=>0, "PauliY"=>0, "Hadamard"=>0, "PauliX"=>0)),
]


# Ordered gate names for plotting (change order here if you prefer)
const GATES = ["PauliZ", "PauliY", "Hadamard", "PauliX"]

# Try to load Plots now (helpful message if missing). This must be at top-level.
try
    using Plots
catch err
    println("Plots.jl is not installed. In the Julia REPL run:\n  using Pkg; Pkg.add(\"Plots\")\nthen re-run this script.")
    rethrow(err)
end

# StatsPlots is optional; detect availability
const HAS_STATSPLOTS = try
    using StatsPlots
    true
catch _
    false
end

function build_matrix(pairs, gates=GATES)
    pairs_sorted = sort(pairs, by = first)
    xs = [p[1] for p in pairs_sorted]
    # matrix with size (n_gates x n_samples)
    mat = [Base.get(p[2], g, 0) for g in gates, p in pairs_sorted]
    return xs, mat
end

function plot_lines(xs, mat, gates; out="gate_flags_lines.png")
    plt = plot(xs, permutedims(mat), dpi = 250,
        labels = gates,
        marker = :circle,
        linewidth = 2,
        xlabel = "x",
        ylabel = "flag (0/1)",
        xscale = :log10,
        yticks = [0, 1],
        ylim = (-0.1, 1.1),
        legend = :topright,
        title = "Gate enabled (1) vs x")
    savefig(plt, out)
    return out
end

function plot_heatmap(xs, mat, gates; out="gate_flags_heatmap.png")
    # gates on x-axis (columns), bias (xs) on y-axis (rows).
    # mat is gates x samples, so permutedims(mat) gives samples x gates which matches (y, x).
    m, n = size(mat)  # m = n_gates, n = n_samples

    # sensible tick labels for the y axis (one per sample)
    tick_labels = ["b₁","b₂","b₃","b₄","b₅","b₆","b₇","b₈"]

    plt = heatmap(gates, xs, permutedims(mat), 
        dpi = 250,
        left_margin   = 1Plots.mm,
        right_margin  = 10Plots.mm,
        top_margin    = 5Plots.mm,
        bottom_margin = 5Plots.mm;
        xlabel = "Gate",
        ylabel = "bias",
        yticks = (xs, reverse(tick_labels[1:length(xs)])),
        yscale = :log10,
        color = cgrad([:blue, :orange]),
        colorbar_ticks = ([0, 1], ["good", "bad"]),
        clims = (0, 1),
        xrotation = 20,
        title = "Gate re-optimization", 
        legend = false
        )

    # add small legend-like annotations indicating color meaning
    y_top = maximum(xs)
    y_bottom = minimum(xs)
    y_spacing_factor = 10.0^(-1/6)   # geometric offset for the second label on log scale
    annotate!(plt, 1.3, y_top * 10^(1/6), text("● re-optimize", :orange, 10, :left))
    annotate!(plt, 2.3, y_top * 10^(1/6), text("● no re-optimize", :blue, 10, :left))

    # draw vertical lines between gate columns to form a grid
    # gates are plotted at integer positions 1:m, so boundaries lie at 0.5, 1.5, ..., m+0.5
    vlines = collect(1:1:m)
    vline!(vlines, color = :black, lw = 0.8, alpha = 0.6, label = false)

    # draw horizontal lines at each sample (xs) to complete the grid
    # hline!(xs./ 2, color = :black, lw = 0.8, alpha = 0.6, label = false)
    hlines = xs .* 10^(-1/12)
    hline!(hlines, color = :black, lw = 0.8, alpha = 0.6, label = false)
    savefig(plt, out)
    return out
end

function main()
    xs, mat = build_matrix(pairs, GATES)

    out1 = plot_lines(xs, mat, GATES)
    println("Wrote line plot to: ", out1)

    out2 = plot_heatmap(xs, mat, GATES)
    println("Wrote heatmap to: ", out2)

    # If StatsPlots was detected, also write grouped bars (optional)
    if HAS_STATSPLOTS
        out3 = "gate_flags_groupedbar.png"
        groupedbar(xs, permutedims(mat), bar_position = :dodge,
                   labels = GATES, xlabel = "x", ylabel = "flag (0/1)",
                   legend = :topright, xscale = :log10)
        savefig(out3)
        println("Wrote grouped bar plot to: ", out3)
    end
end


main()

