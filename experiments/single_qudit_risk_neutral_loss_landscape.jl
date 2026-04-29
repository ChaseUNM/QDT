using ArgParse, LinearAlgebra, Random, Distributions, Printf
using QuantumGateDesign
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
using JLD2
include("../src/digital_qudit.jl")
include("../src/util.jl")



###################################################################
# PARAMETERS
###################################################################

# Device 
Ne      = 2
Ng      = 2
omega   = 4.5
xi      = 0.2 # no xi variance
omega_stdev = 1e-3 * omega
n_samples   = 5

# Gates
gates = [PauliX, PauliY, PauliZ]
n_gates = length(gates)

n_trials = 10 # number of trials to perform

seed = 7286 # rng seed

############################################################## 
# LOAD DATA FROM FILE
##############################################################

data_dir = "./data/single_qudit_risk_neutral_variance"
filename = @sprintf(
    "N%d+%d_stdev%.1e_samples%d_trials%d_seed%d.jld2",
    Ne, Ng, omega_stdev, n_samples, n_trials, seed
)
full_path = joinpath(data_dir, filename)

# Verifying the file exists
if !isfile(full_path)
    error(@sprintf("File not found: %s\n", full_path))
end

# Loading data ...
data = load(full_path)
# Unpacking
degree = data["degree"]
n_splines = data["n_splines"]
T_gate = data["T_gate"]
if isa(T_gate, Float64)
    T_gate = T_gate * ones(n_gates)
end
control_coeffs = data["control_coeffs"]


############################################################## 
# SETUP
##############################################################

# Create a Qudit
q = DigitalQudit(Ne, Ng)
N = Ne + Ng
add_param_samples(q, omega, xi)

# Create a control for each gate
for i = 1:n_gates
    # Control for this gate
    local qcontrol = FortranBSplineControl(degree, n_splines, T_gate[i])
    add_control(q, gates[i], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Vector{Float64})
end


##############################################################
# RUN EXPERIMENT
##############################################################

dt_eval_fidelity = 0.05
n_steps = 85
c = LinRange(-1.05, 1.05, n_steps)

infidelities = zeros(n_gates, n_steps, n_steps)

for g = 1:n_gates
    @printf("\n==== GATE %d of %d ==== \n", g, n_gates)

    # Select the pair of control coefficients with the 
    # smallest overlap
    Beta = control_coeffs[:,g,:]';
    idx = argmin(abs.(Beta'*Beta));
    beta1 = Beta[:,idx.I[1]]
    beta2 = Beta[:,idx.I[2]]
    # Orthogonalize
    beta2 .-= dot(beta1,beta2)/dot(beta1,beta1) * beta1
    # Random direction if beta2 is now too small
    if norm(beta2) < 1e-1
        n = length(beta1)
        beta2 .= norm(beta1) * randn(n) / sqrt(n) 
    end

    # Compute fidelity at each point in the domain
    U = unitary(gates[g], Ne+Ng) 
    for i = 1:n_steps
        for j = 1:n_steps
            # Linear combination of beta 1 and beta 2
            beta = c[i] * beta1 + c[j] * beta2;
            q.controls[gates[g]].coeffs.values[1] = beta
            # Evaluate infidelity
            Psi = run_control(
                    q, 
                    q.controls[gates[g]], 
                    dt=dt_eval_fidelity
                )[1,:,:]
            foreach(normalize!, eachcol(Psi))
            infidelities[g,i,j] = infidelity(Psi, U, Ne)
        end
    end

end


##############################################################
# PLOT RESULTS
##############################################################

fig_dir = "./figures/single_qudit_risk_neutral_variance"
file_prefix = @sprintf(
    "N%d+%d_stdev%.1e_samples%d_trials%d_seed%d",
    Ne, Ng, omega_stdev, n_samples, n_trials, seed
)

subplots = Any[]
gate_names = ["X", "Y", "Z"]
for i = 1:n_gates
    title_str = @sprintf(
                    "Loss Landscape, \$%s\$ Gate", 
                    gate_names[i]
                )
    p_i = heatmap(
            c, c, log10.(infidelities[i,:,:]),
            title= LaTeXString(title_str), 
            titlefontsize=16, 
            xlabel=L"$\eta_1$", xlabelfontsize=14,
            ylabel=L"$\eta_2$", ylabelfontsize=14,
            size=(600,500), dpi=512, 
            left_margin = 10mm, right_margin = 10mm
        )

    push!(subplots, p_i)

    # Save to file
    fig_path = joinpath(fig_dir, 
                        @sprintf("%s_landscape_%s.svg", file_prefix, gate_names[i]))
    savefig(p_i, fig_path)

end

##############################################################
# RUN EXPERIMENT 2: Z-Gate Only, Random Pairs
##############################################################

n_Zpairs = 20
n_steps = 43
c = LinRange(-1.05, 1.05, n_steps)

Beta_Z = control_coeffs[:,3,:]';
infidelities_Z = zeros(n_Zpairs, n_steps, n_steps)

U = unitary(PauliZ, Ne+Ng) 
for p = 1:n_Zpairs
    @printf("\n==== PAIR %d of %d ==== \n", p, n_Zpairs)

    # Select a random pair of controls
    idx = randperm(n_trials)[1:2]
    beta1 = Beta_Z[:,idx[1]]
    beta2 = Beta_Z[:,idx[2]]
    
    # Orthogonalize
    beta2 .-= dot(beta1,beta2)/dot(beta1,beta1) * beta1
    # Random direction if beta2 is now too small
    if norm(beta2) < 1e-1
        n = length(beta1)
        beta2 .= norm(beta1) * randn(n) / sqrt(n) 
    end

    # Compute fidelity at each point in the domain
    for i = 1:n_steps
        for j = 1:n_steps
            # Linear combination of beta 1 and beta 2
            beta = c[i] * beta1 + c[j] * beta2;
            q.controls[PauliZ].coeffs.values[1] = beta
            # Evaluate infidelity
            Psi = run_control(
                    q, 
                    q.controls[PauliZ], 
                    dt=dt_eval_fidelity
                )[1,:,:]
            foreach(normalize!, eachcol(Psi))
            infidelities_Z[p,i,j] = infidelity(Psi, U, Ne)
        end
    end

end



##############################################################
# RUN EXPERIMENT 2: Z-Gate Only, Subplot (β₁, βⱼ)
##############################################################

n_steps = 43
c = LinRange(-1.05, 1.05, n_steps)

infidelities_Z = zeros(n_trials, n_steps, n_steps)

U = unitary(PauliZ, Ne+Ng) 
for t = 2:n_trials
    @printf("\n==== Trial %d of %d ==== \n", t, n_trials)

    # Select a random pair of controls
    beta1 = Beta_Z[:,1]
    beta2 = Beta_Z[:,t]
    
    # Orthogonalize
    beta2 .-= dot(beta1,beta2)/dot(beta1,beta1) * beta1
    # Random direction if beta2 is now too small
    if norm(beta2) < 1e-1
        n = length(beta1)
        beta2 .= norm(beta1) * randn(n) / sqrt(n) 
    end

    # Compute fidelity at each point in the domain
    for i = 1:n_steps
        for j = 1:n_steps
            # Linear combination of beta 1 and beta 2
            beta = c[i] * beta1 + c[j] * beta2;
            q.controls[PauliZ].coeffs.values[1] = beta
            # Evaluate infidelity
            Psi = run_control(
                    q, 
                    q.controls[PauliZ], 
                    dt=dt_eval_fidelity
                )[1,:,:]
            foreach(normalize!, eachcol(Psi))
            infidelities_Z[t,i,j] = infidelity(Psi, U, Ne)
        end
    end
end


# Plotting the loss landscapes over these domains
subplots = Any[]
for t = 2:n_trials
    title_str = @sprintf(
                    "Span{\$\\beta_1\$, \$\\beta_{%d}\$}", 
                    t
                )
    hm = heatmap(
            c, c, log10.(infidelities_Z[t,:,:]),
            title=LaTeXString(title_str),
            xlabel=L"$\eta_1$", xlabelfontsize=14,
            ylabel=L"$\eta_2$", ylabelfontsize=14,
            size=(500,500), dpi=512
        )
    push!(subplots, hm)
end
f = plot(
    subplots..., layout=(3, 3), colorbar=false,
    size=(1000,1000), dpi=512
)
fig_path = joinpath(fig_dir, 
                    @sprintf("%s_landscapes_Z.svg", file_prefix))
savefig(f, fig_path)


##############################################################
# Controls and infidelity over cos(θ) β₁ + sin(n_theta) β₃)
##############################################################

n_theta = 64
n_t = 200
controls = zeros(Float64, n_theta, n_t, 2)
infidelity_vs_theta = zeros(n_theta)

times = LinRange(0, control_obj.tf, n_t)
control_obj = q.controls[PauliZ].objs.values[1]

for i = 1:n_theta
    
    # Set the control vector
    θ = 2*π/n_theta * i
    beta = cos(θ) * Beta_Z[:,1] + sin(θ)*Beta_Z[:,3]
    q.controls[PauliZ].coeffs.values[1] = beta

    # Compute the control signal as a function of time
    controls[i,:,1] = [
            eval_p_derivative(control_obj, time, beta, 0)
            for time in times
        ]
    controls[i,:,2] = [
            eval_q_derivative(control_obj, time, beta, 0)
            for time in times        
    ]

    # Evaluate infidelity
    Psi = run_control(
            q, 
            q.controls[PauliZ], 
            dt=dt_eval_fidelity
        )[1,:,:]
    foreach(normalize!, eachcol(Psi))
    infidelity_vs_theta[i] = infidelity(Psi, U, Ne)
    
end


# Plot real and imaginary parts of the controls
f = plot(
    title="Control Signals in the Degenerate Space",
    xlabel=L"$p(t)$", xlabelfontsize=14,
    ylabel=L"$q(t)$", ylabelfontsize=14,
    size=(500,500), dpi=512,
    legendfontsize=14
)

plot!(
    data["controls"][1,3,1,:],
    data["controls"][1,3,2,:],
    color=:blue, label=L"$\beta_1$",
    linewidth=2
)
plot!(
    data["controls"][3,3,1,:],
    data["controls"][3,3,2,:],
    color=:red, label=L"$\beta_3$",
    linewidth=2
)
for i = [8]
    plot!(
        controls[i,:,1], controls[i,:,2],
        alpha=0.6, color=:black,
        label=""
    )
end
fig_path = joinpath(fig_dir, 
                    @sprintf("%s_controls_Z_degenerate.svg", file_prefix))
savefig(f, fig_path)


# Plot real and imaginary parts of some of the other controls
f = plot(
    title="Other Z Gate Control Signals",
    xlabel=L"$p(t)$", xlabelfontsize=14,
    ylabel=L"$q(t)$", ylabelfontsize=14,
    size=(500,500), dpi=512,
    legendfontsize=14
)

plot!(
    data["controls"][1,3,1,:],
    data["controls"][1,3,2,:],
    color=:blue, label=L"$\beta_1$",
    linewidth=1.5
)
plot!(
    data["controls"][2,3,1,:],
    data["controls"][2,3,2,:],
    label=L"$\beta_2$",
    linewidth=1.5
)
plot!(
    data["controls"][4,3,1,:],
    data["controls"][4,3,2,:],
    label=L"$\beta_4$",
    linewidth=1.5
)
plot!(
    data["controls"][5,3,1,:],
    data["controls"][5,3,2,:],
    label=L"$\beta_5$",
    linewidth=1.5
)
fig_path = joinpath(fig_dir, 
                     @sprintf("%s_controls_Z_polar.svg", file_prefix))
savefig(f, fig_path)


# Compare against the X and Y gate controls
f = plot(
    title="X and Y Controls",
    xlabel=L"$p(t)$", xlabelfontsize=14,
    ylabel=L"$q(t)$", ylabelfontsize=14,
    size=(500,500), dpi=512,
    legendfontsize=14
)
plot!(
    data["controls"][1,1,1,:],
    data["controls"][1,1,2,:],
    label=L"$X$",
    linewidth=1.5
)
plot!(
    data["controls"][1,2,1,:],
    data["controls"][1,2,2,:],
    label=L"$Y$",
    linewidth=1.5
)
fig_path = joinpath(fig_dir, 
                     @sprintf("%s_controls_XY_polar.svg", file_prefix))
savefig(f, fig_path)