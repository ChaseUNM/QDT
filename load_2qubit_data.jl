#load Wasserstein data 

using LinearAlgebra, JLD2, QuantumGateDesign

tag = 28
folder = "results/characterization_2qubits_$tag"
if !isdir(joinpath(folder, "figures"))
    mkpath(joinpath(folder, "figures"))
end
round = 4
@load joinpath(folder, "data", "w2_chain_round$round.jld2") w2_chain_multi mean_data covariance_mat
@load joinpath(folder, "data", "loss_data_round1.jld2") loss ω1_vec ω2_vec ξ12_vec


# plot histogram of w2_chain_multi
ω1_hist = histogram(w2_chain_multi.diagnostic_chain[:,1,1], xlabel = "ω1", ylabel = "Frequency", title = "Histogram of ω1 chain", dpi = 250)
ω2_hist = histogram(w2_chain_multi.diagnostic_chain[:,1,2], xlabel = "ω2", ylabel = "Frequency", title = "Histogram of ω2 chain", dpi = 250)
ξ12_hist = histogram(w2_chain_multi.diagnostic_chain[:,1,3], xlabel = "ξ12", ylabel = "Frequency", title = "Histogram of ξ12 chain", dpi = 250)

savefig(ω1_hist, joinpath(folder, "figures", "ω1_hist_round$round.png"))
savefig(ω2_hist, joinpath(folder, "figures", "ω2_hist_round$round.png"))
savefig(ξ12_hist, joinpath(folder, "figures", "ξ12_hist_round$round.png"))

# plot loss at "optimal" points 

loss_plot_ω1_ω2_heatmap = heatmap(ω1_vec, ω2_vec, log10.(loss[:,:,26]'), xlabel = "ω1", ylabel = "ω2", title = "Loss landscape: ω1 vs ω2 at true ξ12", dpi = 250)
loss_plot_ω1_ω2_surface = surface(ω1_vec, ω2_vec, log10.(loss[:,:,26]'), xlabel = "ω1", ylabel = "ω2", title = "Loss landscape: ω1 vs ω2 at true ξ12", dpi = 250)
loss_plot_ω1_ξ12_heatmap = heatmap(ω1_vec, ξ12_vec, log10.(loss[:,31,:]'), xlabel = "ω1", ylabel = "ξ12", title = "Loss landscape: ω1 vs ξ12 at true ω2", dpi = 250)
loss_plot_ω1_ξ12_surface = surface(ω1_vec, ξ12_vec, log10.(loss[:,31,:]'), xlabel = "ω1", ylabel = "ξ12", title = "Loss landscape: ω1 vs ξ12 at true ω2", dpi = 250)
loss_plot_ω2_ξ12_heatmap = heatmap(ω2_vec, ξ12_vec, log10.(loss[26,:,:]'), xlabel = "ω2", ylabel = "ξ12", title = "Loss landscape: ω2 vs ξ12 at true ω1", dpi = 250)
loss_plot_ω2_ξ12_surface = surface(ω2_vec, ξ12_vec, log10.(loss[26,:,:]'), xlabel = "ω2", ylabel = "ξ12", title = "Loss landscape: ω2 vs ξ12 at true ω1", dpi = 250)

plot_ω1_ω2 = plot(loss_plot_ω1_ω2_heatmap, loss_plot_ω1_ω2_surface, layout = (1, 2))
plot_ω1_ξ12 = plot(loss_plot_ω1_ξ12_heatmap, loss_plot_ω1_ξ12_surface, layout = (1, 2))
plot_ω2_ξ12 = plot(loss_plot_ω2_ξ12_heatmap, loss_plot_ω2_ξ12_surface, layout = (1, 2))


savefig(loss_plot_ω1_ω2_heatmap, joinpath(folder, "figures", "loss_plot_ω1_ω2_heatmap_$round.png"))
savefig(loss_plot_ω1_ω2_surface, joinpath(folder, "figures", "loss_plot_ω1_ω2_surface_$round.png"))
savefig(loss_plot_ω1_ξ12_heatmap, joinpath(folder, "figures", "loss_plot_ω1_ξ12_heatmap_$round.png"))
savefig(loss_plot_ω1_ξ12_surface, joinpath(folder, "figures", "loss_plot_ω1_ξ12_surface_$round.png"))
savefig(loss_plot_ω2_ξ12_heatmap, joinpath(folder, "figures", "loss_plot_ω2_ξ12_heatmap_$round.png"))
savefig(loss_plot_ω2_ξ12_surface, joinpath(folder, "figures", "loss_plot_ω2_ξ12_surface_$round.png"))