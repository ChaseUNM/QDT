using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, LaTeXStrings, JLD2
include("src/digital_qudit.jl")
include("src/physical_device.jl")
include("src/util.jl")

############################################################################# 
# PARAMETERS
#############################################################################

# Device 
Ne = 2
Ng = 0
n_samples = 10


# Controls
degree = 2
n_splines = 6
n_iters_opt = 100

# Physical Qudit
M_spam_order = 1e-4
n_readout_samples = 100000

omega_init = 4.5 
omega_center = omega_init
omega_stdev = 0.001
omega_bias = 0.0

xi_init = 0.0
xi_center = 0.0
xi_stdev = 0.0
xi_bias = 0.0

omega_sampler = Normal(omega_center + omega_bias, omega_stdev)
xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)



############################################################################# 
# SETUP
#############################################################################

# Create a Qudit
q = DigitalQudit(Ne, Ng)
N = Ne + Ng

#Create gate set
gates = [PauliX, PauliY, PauliZ]
N_gates = length(gates)
T_gate = 15 * pi

phys_q = PhysicalQudit(
                    Ne, Ng, 
                    omega_init, omega_rot, xi_init,
                    M_spam_order=M_spam_order
                )

# Create a control for each gate
for i = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[i], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Float64)
end

# Initial parameter samples
omega_samples = sort(rand(omega_sampler, n_samples))
xi_samples = sort(rand(xi_sampler, n_samples))
add_param_samples(q, omega_samples, xi_samples)

omega_rot = omega_init
total_iterations = 100


#Initial optimization 
for j = 1:N_gates
    optimize_control(q, gates[j])
end

state_inf_arr = zeros(N_gates, total_iterations)
pop_inf_arr = zeros(N_gates, total_iterations)

traffic_light_flag = zeros(N_gates, total_iterations)

tol = 1E-3


#Now have our true parameter changing in time and measure fidelity at each time
# for i in 1:total_iterations
#     println("Iteration $i")
#     for j in 1:N_gates 
#         s_inf, p_inf = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples)
#         state_inf_arr[j,i] = s_inf 
#         pop_inf_arr[j,i] = p_inf 
#     end
#     traffic_light_flag[:,i] = pop_inf_arr[:,i] .> tol
#     omega_init += 0.001*abs(randn())
#     xi_init += 0.0
#     add_param_samples(phys_q.qudit, [omega_init], [xi_init]; iter = i+1, average_omega_rot = false)

#     # println("GOt here")
#     # Now need to add flag if p_inf is too small to recharacterize and re-optimize controls
#     # if any(tol .< pop_inf_arr[:,i])
#     #     println("Characterizing more and optimizing again")
#     #     omega_sampler = Normal(omega_init + omega_bias, omega_stdev)
#     #     xi_sampler = Normal(xi_init + xi_bias, xi_stdev)
#     #     omega_samples = sort(rand(omega_sampler, n_samples))
#     #     xi_samples = sort(rand(xi_sampler, n_samples))
#     #     q = DigitalQudit(Ne, Ng)
#     #     N = Ne + Ng
#     #     for k in 1:N_gates 
#     #         optimize_control(q, gates[k])
#     #     end
#     # end

# end

function test(q, phys_q, total_iterations, gates, tol)
    state_inf_arr = zeros(N_gates, total_iterations)
    pop_inf_arr = zeros(N_gates, total_iterations)
    count = 1
    traffic_light_flag = zeros(N_gates, total_iterations)
    omega_init = get(phys_q.qudit.omega)[1]
    xi_init = get(phys_q.qudit.xi)[1]
    for i in 1:total_iterations
        println("Iteration $i")
        for j in 1:N_gates 
            s_inf, p_inf = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples)
            state_inf_arr[j,i] = s_inf 
            pop_inf_arr[j,i] = p_inf 
        end
        traffic_light_flag[:,i] = pop_inf_arr[:,i] .> tol
        omega_init += 0.001*abs(randn())
        xi_init += 0.0
        add_param_samples(phys_q.qudit, [omega_init], [xi_init]; iter = i+1, average_omega_rot = false)
    # println("GOt here")
    # Now need to add flag if p_inf is too small to recharacterize and re-optimize controls
        if any(tol .< pop_inf_arr[:,i])
            count += 1
            println("Characterizing more and optimizing again")
            omega_sampler = Normal(omega_init + omega_bias, omega_stdev)
            xi_sampler = Normal(xi_init + xi_bias, xi_stdev)
            omega_samples = sort(rand(omega_sampler, n_samples))
            xi_samples = sort(rand(xi_sampler, n_samples))

            add_param_samples(q, omega_samples, xi_samples; iter = count)
            N = Ne + Ng
            for k in 1:N_gates 
                println(gates[k])
                optimize_control(q, gates[k])
            end
        end
    end
    return state_inf_arr, pop_inf_arr, traffic_light_flag
end

a, b, c = test(q, phys_q, 100, gates, 1E-3)