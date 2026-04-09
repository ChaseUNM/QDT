using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
include("../src/digital_qudit.jl")
include("../src/util.jl")

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
xi_stdev = 0.01 * xi
n_samples = 50

# Gates
gates = [PauliX, PauliY, PauliZ]
N_gates = length(gates)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 6
n_iters_opt = 100
seed = 3141

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
omega_samples = rand(omega_sampler, n_samples)
xi_samples = rand(xi_sampler, n_samples)
add_param_samples(q, omega_samples, xi_samples)

# Create a control for each gate
for i = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[i], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Vector{Float64})
end

############################################################################# 
# CONTROL OPTIMIZATION
#############################################################################

for i = 1:N_gates
    optimize_control(q, gates[i], 
                        options=["max_iter" => n_iters_opt, "print_level" => 5, "limited_memory_max_history" => 250])
end

infidelities_at_samples = zeros(N_gates, n_samples)
for i = 1:N_gates
    infidelities_at_samples[i,:] = q.infidelity[gates[i]].values[end]
end

############################################################################# 
# INFIDELITY OVER PARAM RANGE
#############################################################################

dt = 0.005
n_pts = 25
domega = 3*0.001*omega
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
        q.omega_rot = mean(omega_samples) # Use the same rotatining frame frequency as the initial set of samples, as this is the frame in which the controls where optimized in
        # Compute the gate fidelity for each of these parameter samples
        Psi_j = run_control(q, q.controls[gates[i]], dt=dt)
        for k = 1:n_pts
            Psi_jk = Psi_j[k,:,:]
            foreach(normalize!, eachcol(Psi_jk))
            infidelities[i,j,k] = infidelity(Psi_jk, U, Ne)
        end
    end
end

c_min = 5e-6
c_max = 1.0
ticks = [1e-5, 1e-3, 1e-1]

# Plot Infidelities vs parameter value
subplots = Any[]
gate_names = ["X", "Y", "Z"]
for i = 1:N_gates
    title_str = @sprintf("Predicted Fidelity, \$%s\$ Gate", gate_names[i])
    p_i = plot(
            title= LaTeXString(title_str), 
            titlefontsize=16, 
            xlabel=L"Frequency Error $(\omega-\omega_*)/\omega_*$", xlabelfontsize=14,
            ylabel=L"Self-Kerr Error $(\xi-\xi_*)/\xi_*$", ylabelfontsize=14, 
            size=(600,500), dpi=512,
            right_margin = 10Plots.mm
        )

    heatmap!(
        (omegas .- omega) ./ omega, 
        (xis .- xi) ./ xi, 
        infidelities[i,:,:]', 
        colorbar_scale=:log10, colorbar_ticks=ticks, 
        clim=(c_min,c_max)
    )

    push!(subplots, p_i)

    filename = @sprintf(
                    "figures/single_qudit_risk_neutral_%s.svg", 
                    gate_names[i]
                )
    # savefig(p_i, filename)

end


plot(
    subplots[1], subplots[2], subplots[3],
    layout=(1,3), size=(1500,400), dpi=512
)