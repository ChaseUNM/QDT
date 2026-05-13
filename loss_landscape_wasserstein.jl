using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, OrderedCollections
include("src/digital_qudit.jl")
include("src/physical_device.jl")
include("src/util.jl")
include("src/wasserstein_inference.jl")
include("src/postprocessing.jl")
include("src/forward_model_quantum.jl")


# get loss landscape of wasserstein distance with forward evolution as parameter varies 
Ne = 2
Ng = 0

ω = 4.5
ωr = 4.5
ξ = 0.0

N_ess = 2
N_guard = 0
N_tot = N_ess + N_guard

H_control_real = [0.0 1;
1 0]
H_control_imag = [0.0 1;
-1 0]
real_control_ops = [H_control_real]
imag_control_ops = [H_control_imag]
degree = 2
n_splines = 10 
T = 50
nsteps = 200
dt = T/nsteps

iter_count = 1

q = DigitalQudit(Ne, Ng)
q.omega_rot = ωr

control = FortranBSplineControl(degree, n_splines, T)
max_control_parameter = 0.1
pcof_l = -max_control_parameter
pcof_u = max_control_parameter
pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter

gates = [PauliX, PauliY, PauliZ, Hadamard]
N_gates = length(gates)
T_gate = 50

delta = ω - ωr
U0 = [1 0; 0 1]
prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)

M_spam_order = 1e-4
n_readout_samples = 100000

phys_q = PhysicalQudit(
                    Ne, Ng, 
                    ω, ωr, ξ,
                    M_spam_order=M_spam_order
                )

println("Creating controls")
for j = 1:N_gates
    # Control for this gate with constant initial pulses
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    pcof0 = 0.5*max_control_parameter*ones(qcontrol.N_coeff)
    
    add_control(q, gates[j], qcontrol; iter = iter_count, coeffs = pcof0)
    # Infidelities for this gate

end

event_obs_total = Vector{Array{Float64}}(undef, 1)
_, event_obs = run_control_physical(phys_q, q.controls[PauliX]; dt = dt)
event_obs = abs2.(event_obs)
M_spam = column_stochastic(1E-4*rand(2))
event_obs = sample_quantum_state_history(100000, M_spam, event_obs)
event_obs_total[1] = event_obs

h = 0.5
pts = 101
param_range = LinRange(ω - h, ω + h, pts)

wasserstein_vec = zeros(pts)




for i in 1:pts 
    ϕ = trace_wasserstein_squared_loss_quantum(param_range[i], ωr, degree, n_splines, U0, T, nsteps, [get(q.controls[PauliX].coeffs)], event_obs_total, 1; δ = 2.0)
    wasserstein_vec[i] = ϕ
end

wasserstein_vec = clamp.(wasserstein_vec, 1E-10, 100)
loss_plot = plot(param_range, wasserstein_vec, xlabel = "ω", ylabel = "ϕ(f, g)", dpi = 250)