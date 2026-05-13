# -------------------- Helpers --------------------

function sample_parameters(omega_center, omega_bias, omega_stdev, xi_center, xi_bias, xi_stdev, n_samples)
    omega_samples = rand(Normal(omega_center + omega_bias, omega_stdev), n_samples)
    xi_samples    = rand(Normal(xi_center + xi_bias, xi_stdev), n_samples)
    return omega_samples, xi_samples
end


function measure_all_gates!(phys_q, q, gates, n_readout_samples)
    pop_inf = Dict{GateType, Float64}()
    state_inf = Dict{GateType, Float64}()

    for g in gates
        s, p = measure_infidelity(phys_q, g, q.controls[g], n_readout_samples)
        pop_inf[g] = p
        state_inf[g] = s
    end

    return state_inf, pop_inf
end


function optimize_if_needed!(q, gates, prev_infidelity, epsilon, iter)
    reoptimized = Dict{GateType, Bool}()

    for g in gates
        needs_opt = prev_infidelity[g] > epsilon
        reoptimized[g] = needs_opt

        if needs_opt
            println("Re-optimizing $g (inf = $(prev_infidelity[g]))")
            optimize_control(q, g; iter=iter, options=["max_iter"=>100, "print_level"=>3])
        end
    end

    return reoptimized
end


function controls_dict(q, gates)
    Dict(g => q.controls[g] for g in gates)
end


# -------------------- Initialization --------------------
# Device 
Ne = 2
Ng = 0
N = Ne + Ng
n_samples = 1000


# Controls
degree = 2
n_splines = 8
n_iters_opt = 100


#Create gate set
gates = [PauliX, PauliY, PauliZ, Hadamard]
N_gates = length(gates)
T_gate = 50


q = DigitalQudit(Ne, Ng)
q.omega_rot = omega_center

omega_samples, xi_samples = sample_parameters(
    omega_center, omega_bias_init, omega_stdev_1,
    xi_center, xi_bias, xi_stdev, n_samples
)

add_param_samples(q, omega_samples, xi_samples; average_omega_rot=false)

# Initialize controls
for g in gates
    add_control(q, g, FortranBSplineControl(degree, n_splines, T_gate))
    q.infidelity[g] = History(Float64)
end

# Initial optimization
for g in gates
    optimize_control(q, g, options=["max_iter"=>100, "print_level"=>3])
end


# Physical Qudit
M_spam_order = 1e-4
n_readout_samples = 100000
true_omega = 4.5
true_xi = 0.0
omega_rot = 4.5
phys_q = PhysicalQudit(
                    Ne, Ng, 
                    true_omega, omega_rot, true_xi,
                    M_spam_order=M_spam_order
                )

# -------------------- Storage --------------------

q_meas_infidelities_pop   = Vector{Dict{GateType, Float64}}()
q_meas_infidelities_state = Vector{Dict{GateType, Float64}}()

recharacterization_flag = Dict{Float64, Bool}()
reoptimization_flag     = Dict{Float64, Dict}()
control_history         = Dict{Float64, Dict}()

# -------------------- First measurement --------------------

state_inf, pop_inf = measure_all_gates!(phys_q, q, gates, n_readout_samples)

push!(q_meas_infidelities_state, state_inf)
push!(q_meas_infidelities_pop, pop_inf)

recharacterization_flag[omega_bias_init] = !all(values(pop_inf) .< epsilon)
control_history[omega_bias_init] = controls_dict(q, gates)

# -------------------- Main loop --------------------

for (iter, bias) in enumerate(omega_bias, start=2)

    omega_samples, xi_samples = sample_parameters(
        omega_center, bias, omega_stdev_1,
        xi_center, xi_bias, xi_stdev, n_samples
    )

    add_param_samples(q, omega_samples, xi_samples; iter=iter, average_omega_rot=false)
    q.omega_rot = omega_center

    prev_inf = q_meas_infidelities_pop[end]

    reopt = optimize_if_needed!(q, gates, prev_inf, epsilon, iter)
    reoptimization_flag[bias] = reopt

    state_inf, pop_inf = measure_all_gates!(phys_q, q, gates, n_readout_samples)

    push!(q_meas_infidelities_state, state_inf)
    push!(q_meas_infidelities_pop, pop_inf)

    recharacterization_flag[bias] = !all(values(pop_inf) .< epsilon)
    control_history[bias] = controls_dict(q, gates)
end