using ArgParse, LinearAlgebra, Random, Distributions, Printf
using QuantumGateDesign
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
using JLD2
include("../src/digital_qudit.jl")
include("../src/util.jl")


###################################################################
# SETUP ARGUMENT PARSER
###################################################################

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ##### QUDIT ####
        "--Ne"
            help = "Number of essential levels"
            arg_type = Int64
            default = 2
        "--Ng"
            help = "Number of guard levels"
            arg_type = Int64
            default = 2
        "--omega"
            help = "Qudit true frequency"
            arg_type = Float64
            default = 4.5
        "--xi"
            help = "Qudit true self-kerr"
            arg_type = Float64
            default = 0.2
        "--omega-stdev"
            help = "Sampling stdev for qudit frequency, as a 
                    proportion of omega"
            arg_type = Float64
            default = 1E-3
        "--samples"
            help = "Number of frequency samples per trial"
            arg_type = Int
            default = 10
        ##### GATES ####
        "--T-gate"
            help = "X and Y gate duration"
            arg_type = Float64
            default = 47.2
        "--T-Zgate"
            help = "Z gate duration"
            arg_type = Float64
            default = 1.5*47.2
        "--control-degree"
            help = "Degree of the splines defining control signals"
            arg_type = Int
            default = 2
        "--control-splines"
            help = "Number of basis functions for the splines defining control signals"
            arg_type = Int
            default = 8
        "--opt-iters"
            help = "Maximum number of iterations for IPOPT control signal optimization"
            arg_type = Int
            default = 100
        ##### MISC ####
        "--trials"
            help = "Number of independent trials to perform"
            arg_type = Int
            default = 2
        "--n-omega-eval"
            help = "Number of frequencies used to estimate gate infidelity distribution post optimization"
            arg_type = Int
            default = 1024
        "--seed"
            help = "Seed for the global RNG"
            arg_type = Int
            default = -1 # Selected randomly
    end

    return parse_args(s)
end

###################################################################
# PARSE ARGUMENTS
###################################################################

function main()

    t_main_start = time()

    parsed_args = parse_commandline()

    # Device 
    Ne      = parsed_args["Ne"]
    Ng      = parsed_args["Ng"]
    omega   = parsed_args["omega"]
    xi      = parsed_args["xi"] # no xi variance
    omega_stdev = parsed_args["omega-stdev"] * omega
    n_samples   = parsed_args["samples"]

    # Gates
    gates = [PauliX, PauliY, PauliZ]
    n_gates = length(gates)
    T_XYgates  = parsed_args["T-gate"]
    T_Zgate = parsed_args["T-Zgate"]
    T_gate = [T_XYgates T_XYgates T_Zgate]

    # Controls
    degree    = parsed_args["control-degree"]
    n_splines = parsed_args["control-splines"]
    opt_iters = parsed_args["opt-iters"]

    n_trials = parsed_args["trials"] # number of trials to perform

    seed = parsed_args["seed"] # rng seed
    if seed < 1
        seed = Int(rand(UInt16))
    end

    # Range of omega values over which to eval fidelity
    n_omega_eval = parsed_args["n-omega-eval"]
    d_omega = 2 * omega_stdev  
    omega_min = omega - d_omega
    omega_max = omega + d_omega
    dt_eval_fidelity = 0.01

    # Number of times at which to evaluate the control functions
    # for plotting
    n_t_eval = 256


    ############################################################## 
    # SETUP
    ##############################################################

    # Set the seed for the RNG
    Random.seed!(seed)

    # Frequency parameter sampler
    omega_sampler = Normal(omega, omega_stdev)

    # Range of omegas over which to eval infidelities
    omega_range = collect(LinRange(omega_min,omega_max,n_omega_eval))

    # Larger set of omega samples used to estimate infidelity distributions
    omega_large_sample_set = sort(rand(omega_sampler, n_omega_eval))

    # Combine the above into a single set of omegas
    # so we can more easily evaluate fidelities on it
    omegas_eval = [omega_range; omega_large_sample_set]
    xi_eval = xi * ones(2*n_omega_eval) # No variance in self-kerr

    # Create a Qudit
    q = DigitalQudit(Ne, Ng)
    N = Ne + Ng
    add_param_samples(q, omega, xi) # Initializes the param hist

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

    controls = zeros(n_trials, n_gates, 2, n_t_eval)
    control_coeffs = zeros(n_trials, n_gates, degree*n_splines)
    infidelities = zeros(n_trials, n_gates, 2*n_omega_eval)
    omega_samples = zeros(n_trials, n_samples)

    for t = 1:n_trials

        t_start = time()
        @printf("\n==== TRIAL %d of %d ==== \n", t, n_trials)

        # Parameter samples
        omega_samples[t,:] .= sort(rand(omega_sampler, n_samples))
        xi_samples = xi * ones(n_samples)
        update_param_samples(q, omega_samples[t,:], xi_samples)

        for i = 1:n_gates
            @printf("  Optimizing Gate %d of %d\n", i, n_gates)
            # Randomize the controls
            randomize_coeffs(q.controls[gates[i]])
            # Optimize the gate
            optimize_control(q, gates[i], 
                                options=["max_iter" => opt_iters, "print_level" => 0, "limited_memory_max_history" => 250])
            # Plot the control signal in the rotating frame
            local control_obj = q.controls[gates[i]].objs.values[end]
            control_coeffs[t,i,:] = q.controls[gates[i]].coeffs.values[end]
            controls[t,i,1,:] = [
                    eval_p_derivative(
                        control_obj, time, control_coeffs[t,i,:], 0
                    )
                    for time in LinRange(0, T_gate[i], n_t_eval)
                ]
            controls[t,i,2,:] = [
                    eval_q_derivative(
                        control_obj, time, control_coeffs[t,i,:], 0
                    )
                    for time in LinRange(0, T_gate[i], n_t_eval)
                ]
        end

        # Set the parameter samples to be evenly spaced between
        # omega_min and omega_max
        update_param_samples(q, omegas_eval, xi_eval)
        q.omega_rot = mean(omega_samples[t,:]) # Use the same rotatining frame frequency as the initial set of samples, as this is the frame in which the controls where optimized

        # Measure fidelity over grid of omega values
        # and the larger set of samples
        for i = 1:n_gates
            @printf("  Evaluating Gate %d of %d\n", i, n_gates)
            U = unitary(gates[i], Ne+Ng) 
            Psi = run_control(
                    q, 
                    q.controls[gates[i]], 
                    dt=dt_eval_fidelity
                )
            for k = 1:(2*n_omega_eval)
                Psi_k = Psi[k,:,:]
                foreach(normalize!, eachcol(Psi_k))
                infidelities[t,i,k] = infidelity(Psi_k, U, Ne)
            end
        end
        t_end = time()
        @printf("Done. Trial Runtime = %.2f min\n", 
                (t_end-t_start)/60)
        @printf("Program Elapsed Time = %.2f min\n", 
                (t_end-t_main_start)/60)
        
    end

    # Split infidelities array into two pieces
    infidelity_vs_omega = infidelities[:,:,1:n_omega_eval]
    infidelity_samples    = infidelities[:,:,n_omega_eval+1:end]

    ##############################################################
    # SAVE DATA TO FILE
    ##############################################################

    output_dir = "./data/single_qudit_risk_neutral_variance"
    mkpath(output_dir)

    filename = @sprintf(
        "N%d+%d_stdev%.1e_samples%d_trials%d_seed%d.jld2",
        Ne, Ng, omega_stdev, n_samples, n_trials, seed
    )
    full_path = joinpath(output_dir, filename)

    jldsave(full_path; 
            Ne, Ng, omega, xi,  
            T_gate, degree, n_splines,  
            omega_stdev, n_samples, omega_samples, seed, 
            controls, control_coeffs,
            omega_range, infidelity_vs_omega,
            omega_large_sample_set, infidelity_samples,  
    )


    ##############################################################
    # PLOT INFIDELITIES VS OMEGA
    ##############################################################

    c_min = 1e-5
    c_max = 1.0

    # Plot Infidelities vs parameter value
    subplots = Any[]
    gate_names = ["X", "Y", "Z"]
    for i = 1:n_gates
        title_str = @sprintf(
                        "Infidelity by Trial, \$%s\$ Gate", 
                        gate_names[i]
                    )
        p_i = plot(
                title= LaTeXString(title_str), 
                titlefontsize=16, 
                xlabel=L"Frequency Error $(\omega-\omega_*)/\omega_*$", xlabelfontsize=14,
                ylabel="Digital Gate Infidelity", ylabelfontsize=14, 
                size=(600,500), dpi=512, 
                ylim=(c_min,c_max), yscale=:log10,
                right_margin = 10mm
            )
        for t = 1:n_trials
            plot!(
                (omega_range .- omega) ./ omega,
                infidelity_vs_omega[t,i,:],
                color=:blue, alpha=0.5, legend=false
            )
        end

        push!(subplots, p_i)

    end

    f1 = plot(
        subplots[1], subplots[2], subplots[3],
        layout=(1,3), size=(1500,500), dpi=512, 
        bottom_margin=10mm, top_margin=10mm
    )

    # Create figure output directory
    fig_dir = "./figures/single_qudit_risk_neutral_variance"
    mkpath(fig_dir)

    # Saving to file
    file_prefix = @sprintf(
        "N%d+%d_stdev%.1e_samples%d_trials%d_seed%d",
        Ne, Ng, omega_stdev, n_samples, n_trials, seed
    )
    fig_path = joinpath(fig_dir, file_prefix * "_infidelity_vs_omega.svg")
    savefig(f1, fig_path)


    ##############################################################
    # PLOT INFIDELITIES HISTOGRAM
    ##############################################################

    # Histogram of infidelities by gate
    subplots = Any[]
    for i = 1:n_gates
        title_str = @sprintf(
                        "Infidelity by Trial, \$%s\$ Gate", 
                        gate_names[i]
                    )
        p_i = plot(
                title= LaTeXString(title_str), 
                titlefontsize=16, 
                xlabel=L"Log Gate Infidelity $\log_{10}[\mathcal{I}]$", xlabelfontsize=14,
                ylabel=L"Est. Density $\pi(\log_{10}[\mathcal{I}])$", ylabelfontsize=14, 
                size=(600,500), dpi=512, 
                right_margin = 10mm
            )
        for t = 1:n_trials
            stephist!(
                log10.(infidelity_samples[t,i,:]),
                normalize=:pdf, legend=false,
                linewidth=2.5, alpha=0.5
            )
        end

        push!(subplots, p_i)

    end

    f = plot(
        subplots[1], subplots[2], subplots[3],
        layout=(1,3), size=(1500,500), dpi=512, 
        bottom_margin=10mm, top_margin=10mm
    )

    # Saving to file
    fig_path = joinpath(fig_dir, file_prefix * "_infidelity_distribution.svg")
    savefig(f, fig_path)



    ##############################################################
    # PLOT CONTROL SIGNALS
    ##############################################################

    # Plot Infidelities vs parameter value
    control_plots = Any[]
    gate_names = ["X", "Y", "Z"]
    for i = 1:n_gates
        title_str = @sprintf(
                        "Controls by Trial, \$%s\$ Gate", 
                        gate_names[i]
                    )
        p_i = plot(
                title= LaTeXString(title_str), 
                titlefontsize=16, 
                xlabel=L"Time $t$", xlabelfontsize=14,
                ylabel="Control Amplitude, p(t) or q(t)", ylabelfontsize=14, 
                size=(600,500), dpi=512, 
                ylim=(-0.1,0.1),
                right_margin = 10Plots.mm
            )
        for t = 1:n_trials
            plot!(
                LinRange(0, T_gate[i], n_t_eval),
                controls[t,i,1,:],
                color=:blue, alpha=0.5, legend=false
            )
            plot!(
                LinRange(0, T_gate[i], n_t_eval),
                controls[t,i,2,:],
                color=:red, alpha=0.5, legend=false
            )
        end

        push!(control_plots, p_i)

    end


    f2 = plot(
        control_plots[1], control_plots[2], control_plots[3],
        layout=(1,3), size=(1500,500), dpi=512,
        bottom_margin=10mm, top_margin=10mm
    )

    fig_path = joinpath(fig_dir, file_prefix * "_controls.svg")
    savefig(f2, fig_path)

end # end main()


main()