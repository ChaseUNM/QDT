using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
include("../src/digital_device.jl")

############################################################################# 
# PARAMETERS
#############################################################################

# Device 
Ne = 2
Ng = 2
omega = 4.5
omega_bias = 0.002
omega_stdev = 0.001 * omega
xi = 0.2
xi_bias = 0.0
xi_stdev = 0.001 * xi
n_samples = 100

# Gates
gates = [PauliX, PauliY, PauliZ]
N_gates = length(gates)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 6
n_iters_opt = 100
seed = 3141

# Physical Qudit
M_spam_order = 1e-3
n_readout_samples = 2000

############################################################################# 
# SETUP
#############################################################################

# Create parameter samplers
omega_sampler = Normal(omega + omega_bias, omega_stdev)
xi_sampler = Normal(xi + xi_bias, xi_stdev)

# Create a Qudit
q = DigitalQudit(Ne, Ng)
N = Ne + Ng

# Initial parameter samples
omega_samples = sort(rand(omega_sampler, n_samples))
xi_samples = sort(rand(xi_sampler, n_samples))
add_param_samples(q, omega_samples, xi_samples)

# Create a control for each gate
for i = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[i], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Float64)
end

############################################################################# 
# CONTROL OPTIMIZATION
#############################################################################

for i = 1:N_gates
    optimize_control(q, gates[i])
end

# Fidelity for each parameter sample
dt = 0.005
predicted_infidelity = zeros(N_gates, n_samples)
for i = 1:N_gates
    Psi = run_control(q, q.controls[gates[i]], dt=dt)
    U = unitary(gates[i], Ne+Ng)
    for j = 1:n_samples
        Psi_j = Psi[j,:,:]
        foreach(normalize!, eachcol(Psi_j))
        predicted_infidelity[i,j] = infidelity(Psi_j, U, Ne)
    end
end

############################################################################# 
# INFIDELITY OVER PARAM RANGE
#############################################################################

dt = 0.005
n_pts = 25
domega = 3*omega_stdev
dxi = xi/10

omegas = collect(LinRange(omega-domega,omega+domega,n_pts))
xis = collect(LinRange(xi-dxi,xi+dxi,n_pts))

infidelities = zeros(N_gates, n_pts, n_pts)
for i = 1:N_gates
    @printf("Gate %d of %d\n", i, N_gates)
    # Target unitary
    U = unitary(gates[i], Ne+Ng) 
    for j = 1:n_pts
        # Set the digital device to use parameters (omega[j], xi[k]), k = 1, ..., n_pts
        omega_j = omegas[j] * ones(n_pts)
        update_param_samples(q, omega_j, xis)
        q.omega_rot = mean(omega_samples)
        # Compute the gate fidelity for each of these parameter samples
        Psi_j = run_control(q, q.controls[gates[i]], dt=dt)
        for k = 1:n_pts
            Psi_jk = Psi_j[k,:,:]
            foreach(normalize!, eachcol(Psi_jk))
            infidelities[i,j,k] = infidelity(Psi_jk, U, Ne)
        end
    end
end

c_min = minimum(infidelities)
c_max = maximum(infidelities)
ticks = [1e-6, 1e-4, 1e-2, 1e0]

# Plot Infidelities vs parameter value
subplots = Any[]
gate_names = ["X", "Y", "Z"]
for i = 1:N_gates
    title_str = @sprintf("Predicted Fidelity, \$%s\$ Gate", gate_names[i])
    p_i = plot(
            title= LaTeXString(title_str), 
            titlefontsize=16, 
            xlabel=L"Frequency $\omega$", xlabelfontsize=14,
            ylabel=L"Self-Kerr $\xi$", ylabelfontsize=14, 
            size=(500,450), dpi=512,
            right_margin = 10Plots.mm
        )

    heatmap!(
        omegas, 
        xis, 
        infidelities[i,:,:]', 
        colorbar_scale=:log10, colorbar_ticks=ticks, 
        clim=(c_min,c_max)
    )

    push!(subplots, p_i)

end


plot(
    subplots[1], subplots[2], subplots[3],
    layout=(1,3), size=(1500,400), dpi=512
)