using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2
include("src/digital_qudit.jl")
include("src/physical_device.jl")
include("src/util.jl")

# Device 
Ne = 2
Ng = 0
N = Ne + Ng
n_samples = 10


# Controls
degree = 2
n_splines = 10
n_iters_opt = 100


#Create gate set
gates = [PauliX, PauliY, PauliZ, Hadamard]
N_gates = length(gates)
T_gate = 100


##############################################################################
# PREDICTED FIDELITIES DIFFER 
##############################################################################



omega_center = 4.5
omega_stdev_1 = 0.001
omega_stdev_2 = 0.005
omega_bias_1 = 0.1
omega_bias_2 = 0.0

xi_center = 0.0
xi_stdev = 0.0
xi_bias = 0.0

# Create a Qudit
# q_1 = DigitalQudit(Ne, Ng)
# q_2 = DigitalQudit(Ne, Ng)
# q_1.omega_rot = omega_center
# q_2.omega_rot = omega_center

# omega_sampler_q1 = Normal(omega_center + omega_bias_1, omega_stdev_1)
# omega_sampler_q2 = Normal(omega_center + omega_bias_2, omega_stdev_2)
# xi_sampler_q1    = Normal(xi_center + xi_bias, xi_stdev)
# xi_sampler_q2    = Normal(xi_center + xi_bias, xi_stdev*0.5)
# omega_samples_q1 = rand(omega_sampler_q1, n_samples)
# omega_samples_q2 = rand(omega_sampler_q2, n_samples)
# xi_samples_q1    = rand(xi_sampler_q1, n_samples)
# xi_samples_q2    = rand(xi_sampler_q2, n_samples)
# add_param_samples(q_1, omega_samples_q1, xi_samples_q1; average_omega_rot = false)
# add_param_samples(q_2, omega_samples_q2, xi_samples_q2; average_omega_rot = false)



# # Create a control for each gate
# for i = 1:N_gates
#     # Control for this gate
#     qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
#     add_control(q_1, gates[i], qcontrol)
#     add_control(q_2, gates[i], qcontrol)
#     # Infidelities for this gate
#     q_1.infidelity[gates[i]] = History(Float64)
#     q_2.infidelity[gates[i]] = History(Float64)
# end


# #Initial optimization 
# for j = 1:N_gates
#     optimize_control(q_1, gates[j])
#     optimize_control(q_2, gates[j])
# end

# println("Predicted infidelities: ")
# println("Qubit 1: ")
# for i in 1:N_gates
#     println(gates[i])
#     println(get(q_2.infidelity[gates[i]]))
# end

# println("Qubit 2: ")
# for i in 1:N_gates 
#     println(gates[i])
#     println(get(q_1.infidelity[gates[i]]))
# end
#############################################################################
# GET MEASURED INFIDELITY 
#############################################################################

# Physical Qudit
M_spam_order = 1e-4
n_readout_samples = 100000
# n_pts = 1001
true_omega = 4.5
true_xi = 0.0
omega_rot = 4.5
phys_q = PhysicalQudit(
                    Ne, Ng, 
                    true_omega, omega_rot, true_xi,
                    M_spam_order=M_spam_order
                )

# q1_p_meas_infidelity = Dict{GateType, Float64}()
# q2_p_meas_infidelity = Dict{GateType, Float64}()

# q1_s_meas_infidelity = Dict{GateType, Float64}()
# q2_s_meas_infidelity = Dict{GateType, Float64}()

# for i in 1:N_gates
#     q1_s_inf, q1_p_inf = measure_infidelity(phys_q, gates[i], q_1.controls[gates[i]], n_readout_samples)
#     q2_s_inf, q2_p_inf = measure_infidelity(phys_q, gates[i], q_2.controls[gates[i]], n_readout_samples)
#     q1_p_meas_infidelity[gates[i]] = q1_p_inf 
#     q2_p_meas_infidelity[gates[i]] = q2_p_inf 
#     q1_s_meas_infidelity[gates[i]] = q1_s_inf 
#     q2_s_meas_infidelity[gates[i]] = q2_s_inf 
# end

# println("Qubit 1 measured infidelities: ")
# println(q1_p_meas_infidelity)
# println("Qubit 2 measured infidelities: ")
# println(q2_p_meas_infidelity)

# #Plot measured and predicted fidelities 
# predicted_infidelities_q1 = [get(q_1.infidelity[gates[i]]) for i in 1:N_gates]
# predicted_infidelities_q2 = [get(q_2.infidelity[gates[i]]) for i in 1:N_gates]
# measured_infidelities_q1 = [q1_p_meas_infidelity[gates[i]] for i in 1:N_gates]
# measured_infidelities_q2 = [q2_p_meas_infidelity[gates[i]] for i in 1:N_gates]

# epsilon = 1E-4
# infidelity_plot = scatter(1:N_gates, predicted_infidelities_q1, label = "Pred: ω ∈ N(μ + b, 0.005)", yscale=:log10, xlabel = "Gate", ylabel = "Infidelity", xticks=(1:N_gates, ["X", "Y", "Z", "H"]), yticks = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], dpi=256, legend_background_color=:transparent, marker =:circle, legend =:bottomleft)
# scatter!(1:N_gates, predicted_infidelities_q2, label = "Pred: ω ∈ N(μ, 0.005)", yscale=:log10, xlabel = "Gate", ylabel = "Infidelity", xticks=(1:N_gates, ["X", "Y", "Z", "H"]), marker =:circle)
# scatter!(1:N_gates, measured_infidelities_q1, label = "Meas: ω ∈ N(μ + b, 0.005)", yscale=:log10, xlabel = "Gate", ylabel = "Infidelity", xticks=(1:N_gates, ["X", "Y", "Z", "H"]), marker =:square)
# scatter!(1:N_gates, measured_infidelities_q2, label = "Meas: ω ∈ N(μ, 0.005)", yscale=:log10, xlabel = "Gate", ylabel = "Infidelity", xticks=(1:N_gates, ["X", "Y", "Z", "H"]), marker =:square)
# hline!([epsilon], label = "Infidelity Tolerance", color=:black, linestyle=:dash)
# # savefig(infidelity_plot, "red_light_infidelity_plot.png")

# #Get histogram of parameters used for q_1 and q_2 
# q_1_samples = get(q_1.omega)
# q_2_samples = get(q_2.omega)

# hist_q_1 = histogram(q_1_samples, xlabel = "ω", ylabel = "counts", label = "Distribution", dpi = 250)
# vline!([omega_true], label = "true ω₁")

# hist_q_2 = histogram(q_2_samples, xlabel = "ω", ylabel = "counts", label = "Distribution", dpi = 250)
# vline!([omega_true], label = "true ω₁")

epsilon = 1E-4
#Do the entire process above but loop for different bias values and stdev values to see how the predicted and measured infidelities change with different parameter distributions.
omega_bias = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0]
std_dev_init = 0.1

omega_bias_all = vcat(10 .^ -LinRange(1, 5, 25), 0)
omega_bias = omega_bias_all[2:end]
q_meas_infidelities_state = Vector{Dict{GateType, Float64}}(undef, length(omega_bias) + 1)
q_meas_infidelities_pop = Vector{Dict{GateType, Float64}}(undef, length(omega_bias) + 1)
q_list = Vector{DigitalQudit}(undef, length(omega_bias) + 1)
count = 1

omega_bias_init = omega_bias_all[1]

omega_sampler = Normal(omega_center + omega_bias_init, omega_stdev_1)
xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)
omega_samples = rand(omega_sampler, n_samples)
xi_samples    = rand(xi_sampler, n_samples)

q = DigitalQudit(Ne, Ng)
q.omega_rot = omega_center
add_param_samples(q, omega_samples, xi_samples; average_omega_rot = false)
q_p_meas_infidelity = Dict{GateType, Float64}()
q_s_meas_infidelity = Dict{GateType, Float64}()

recharacterization_flag = Dict{Float64, Bool}() 
reoptimization_flag = Dict{Float64, Dict}()

control_history = Dict{Float64, Dict}()

# Create a control for each gate
for j = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[j], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[j]] = History(Float64)
end
control_dict = Dict{GateType, QuditControl}()
for j = 1:N_gates
    optimize_control(q, gates[j], options=["max_iter" => 100, "print_level" => 3])
    control_dict[gates[j]] = q.controls[gates[j]]
end    
control_history[omega_bias_init] = control_dict
q_list[count] = q
for i in 1:N_gates
    q_s_inf, q_p_inf = measure_infidelity(phys_q, gates[i], q.controls[gates[i]], n_readout_samples)
    q_p_meas_infidelity[gates[i]] = q_p_inf 
    q_s_meas_infidelity[gates[i]] = q_s_inf 
end
q_meas_infidelities_state[count] = q_s_meas_infidelity
q_meas_infidelities_pop[count] = q_p_meas_infidelity

if all(values(q_meas_infidelities_pop[count]) .< epsilon)
    println("All measured fidelities sufficiently small")
    recharacterization_flag[omega_bias_init] = false 
else
    println("Need more characterization")
    recharacterization_flag[omega_bias_init] = true
end

count += 1


for i in omega_bias
    # Create a Qudit

    omega_sampler = Normal(omega_center + i, omega_stdev_1)
    xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)
    omega_samples = rand(omega_sampler, n_samples)
    xi_samples    = rand(xi_sampler, n_samples)
    add_param_samples(q, omega_samples, xi_samples; iter = count, average_omega_rot = false)
    q.omega_rot = omega_center
    q_p_meas_infidelity = Dict{GateType, Float64}()
    q_s_meas_infidelity = Dict{GateType, Float64}()
    # for j = 1:N_gates
    #     # Control for this gate
    #     qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    #     add_control(q, gates[j], qcontrol; iter = count)
    #     # Infidelities for this gate
    #     q.infidelity[gates[j]] = History(Float64)
    # end
    # Optimization if previous infidelity wasn't small enough
    # If infidelity was small enough, then won't need to re-optimize
    optim_dict = Dict{GateType, Bool}()
    control_dict = Dict{GateType, QuditControl}()
    for j = 1:N_gates
        if q_meas_infidelities_pop[count - 1][gates[j]] > epsilon
            optim_dict[gates[j]] = true
            println("#######################################################################")
            println("infidelity too large, re-optimizing: ", q_meas_infidelities_pop[count - 1][gates[j]])
            println("#######################################################################")

            optimize_control(q, gates[j]; iter = count, options=["max_iter" => 100, "print_level" => 3])
        else
            optim_dict[gates[j]] = false
            println("########################################################################")
            println("small enough infidelity, no need to re-optimize (hopefully): ", q_meas_infidelities_pop[count - 1][gates[j]])
            println("#########################################################################")
        end
        control_dict[gates[j]] = q.controls[gates[j]]
    end
    # q_list[count] = q
    reoptimization_flag[i] = optim_dict
    for i in 1:N_gates
        q_s_inf, q_p_inf = measure_infidelity(phys_q, gates[i], q.controls[gates[i]], n_readout_samples)
        q_p_meas_infidelity[gates[i]] = q_p_inf 
        q_s_meas_infidelity[gates[i]] = q_s_inf 
    end
    q_meas_infidelities_state[count] = q_s_meas_infidelity
    q_meas_infidelities_pop[count] = q_p_meas_infidelity
    
    if all(values(q_meas_infidelities_pop[count]) .< epsilon)
        println("All measured fidelities sufficiently small")
        recharacterization_flag[i] = false 
    else
        println("Need more characterization")
        recharacterization_flag[i] = true
    end
    control_history[i] = control_dict
    global count += 1
end
   
#Save q_list and q_meas_infidelities_state and q_meas_infideltities_pop
# save_object("q_list.jld2", q_list)
# save_object("q_meas_infidelities_state.jld2", q_meas_infidelities_state)
# save_object("q_meas_infidelities_pop.jld2", q_meas_infidelities_pop)
# save_object("recharacterization_flag.jld2", recharacterization_flag)
# save_object("reoptimization_flag.jld2", reoptimization_flag)

# #loading object 
# q_list = load_object("q_list.jld2")
# q_meas_infidelities_state = load_object("q_meas_infidelities_state.jld2")
# q_meas_infidelities_pop = load_object("q_meas_infidelities_pop.jld2")


#Plot q_meas_infidelities_pop as a scatter plot

#Get the first value that equals 1 in the dictionary 

sorted_dict = sort(collect(recharacterization_flag); by = x -> x.first, rev = true)
pair = findfirst(p -> p.second == 0, sorted_dict)

sorted_reoptimization = sort(collect(reoptimization_flag); by = x -> x.first, rev = true)

measured_infidelities_all_qubits = []
for i in 1:pair
    q_inf_dict = q_meas_infidelities_pop[i]
    q_inf_array = [q_inf_dict[gates[j]] for j in 1:N_gates]
    push!(measured_infidelities_all_qubits, q_inf_array)
end

# Shorter legend (show only first, middle, last), jittered x for readability, nicer colors/markers
bias_vals = [omega_bias_init; omega_bias][1:pair]
n_bias = length(bias_vals)

# get a distinct color for each bias using a perceptually-uniform palette
colors = palette(:viridis, n_bias)

jitter = 0.08
mid = ceil(Int, n_bias/2)

for idx in 1:n_bias
    x = (1:N_gates) .+ (idx - (n_bias+1)/2) * jitter
    b_str = "b"*int_to_subscript(idx)
    # lbl_str = string(b_str, ": ", round(bias_vals[idx]; sigdigits=3))
    # lbl = idx in (1, mid, n_bias) ? lbl_str : nothing

    if idx == 1
        meas_infidelity_scatter = scatter(
            x,
            measured_infidelities_all_qubits[idx],
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Measured population infidelity",
            xticks = (1:N_gates, ["X", "Y", "Z", "H"]),
            yticks = 10.0 .^ (-7:0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = colors[idx],
            label = b_str,
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
    else
        scatter!(
            x,
            measured_infidelities_all_qubits[idx],
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = colors[idx],
            label = b_str
        )
    end
end

hline!([epsilon], label = "Tolerance", color = :black, linestyle = :dash, alpha = 0.8)
# grid! isn't defined in this scope; enable the grid via plot! on the existing plot.
# Use a try/catch in case the scatter object isn't available.
try
    plot!(meas_infidelity_scatter, grid = true)
catch
    plot!(grid = true)
end

# for i in 2:length(q_meas_infidelities_pop)
#     println(measured_infidelities_all_qubits[i])
# end


# savefig(meas_infidelity_scatter, "yellow_light_population_infidelity_scatter.png")

# -------------------- Prepare data --------------------

# bias_vals = [omega_bias_init; omega_bias][1:pair]

# gate_labels = ["X", "Y", "Z", "H"]

# # Color map for bias
# cmap = cgrad(:viridis, length(bias_vals))

# # -------------------- Base plot --------------------

# meas_infidelity_scatter = scatter(
#     1:N_gates,
#     measured_infidelities_all_qubits[1],
#     yscale = :log10,
#     xlabel = "Gate",
#     ylabel = "Measured population infidelity",
#     xticks = (1:N_gates, gate_labels),
#     yticks = 10.0 .^ (-7:0),
#     dpi = 300,
#     legend = false,
#     marker = :circle,
#     color = cmap[1],
#     markersize = 5,
#     alpha = 0.8
# )

# # -------------------- Overlay all bias cases --------------------

# jitter = 0.08

# for (i, b) in enumerate(bias_vals)
#     x = (1:N_gates) .+ (i - length(bias_vals)/2) * jitter

#     scatter!(
#         x,
#         measured_infidelities_all_qubits[i],
#         color = cmap[i],
#         marker = :circle,
#         markersize = 5,
#         alpha = 0.8,
#         label = nothing
#     )
# end

# # -------------------- Threshold line --------------------

# hline!(
#     [epsilon],
#     linestyle = :dash,
#     color = :black,
#     alpha = 0.6,
#     label = "Tolerance"
# )

# # -------------------- Colorbar for bias --------------------

# colorbar!(
#     bias_vals,
#     label = "Bias"
# )



# for i in 1:n_pts
    
#     omega_rot = true_omega

#     true_xi = 0.0


#     phys_q = PhysicalQudit(
#                         Ne, Ng, 
#                         true_omega_range[i], omega_rot, true_xi,
#                         M_spam_order=M_spam_order
#                     )

#     q1_p_meas_infidelity = Dict{GateType, Float64}()
#     q2_p_meas_infidelity = Dict{GateType, Float64}()

#     q1_s_meas_infidelity = Dict{GateType, Float64}()
#     q2_s_meas_infidelity = Dict{GateType, Float64}()

#     for i in 1:N_gates
#         q1_s_inf, q1_p_inf = measure_infidelity(phys_q, gates[i], q_1.controls[gates[i]], n_readout_samples)
#         q2_s_inf, q2_p_inf = measure_infidelity(phys_q, gates[i], q_2.controls[gates[i]], n_readout_samples)
#         q1_p_meas_infidelity[gates[i]] = q1_p_inf 
#         q2_p_meas_infidelity[gates[i]] = q2_p_inf 
#         q1_s_meas_infidelity[gates[i]] = q1_s_inf 
#         q2_s_meas_infidelity[gates[i]] = q2_s_inf 
#         if q1_p_inf > 1E-4
#             println("True Omega: ", true_omega_range[i])
#             println("Qubit 1 infidelity greater than tolerance: good!")
#         end 
        
#         if q2_p_inf > 1E-4 
#             println("True Omega: ", true_omega_range[i])
#             println("Qubit 2 infidelity greater than tolerance: bad!")
#         end
#     end
# end

# helper to convert integer to unicode subscript
_subdigits = Dict('0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄','5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉')
function int_to_subscript(n::Integer)
    s = string(n)
    out = IOBuffer()
    for c in s
        print(out, Base.get(_subdigits, c, c))
    end
    return String(take!(out))
end

#Plot each historgram 

#param sample data 

# plotting_indices = [1,4,8]

# param_sample_data = Vector{Vector{Float64}}(undef, length(omega_bias_all)) 
# param_sample_data[1] = first(q.omega)[2]
# iter_vector = ValueHistories.get(q.omega)[1]
# histogram_plot = histogram(param_sample_data[1], xlabel = "ω", ylabel = "counts", dpi = 250, label = "b₁", legend_background_color=:transparent, legend_columns = 4, legend=:outertop, color = colors[plotting_indices[1]], alpha = 0.8)





# for i in collect(2:length(plotting_indices))
#     # println(typeof(q.omega))
#     println("i: $i")
#     println("plotting_indices[i]: $(plotting_indices[i])")
#     param_sample_data[plotting_indices[i]] = get(q.omega, plotting_indices[i])
#     label_str = "b" * int_to_subscript(plotting_indices[i])
#     println(label_str)
#     histogram!(param_sample_data[plotting_indices[i]], xlabel = "ω", ylabel = "counts", dpi = 250, label = label_str, color = colors[plotting_indices[i]], alpha = 0.8)
# end

# vline!([omega], label = "true ω₁", lw = 3)