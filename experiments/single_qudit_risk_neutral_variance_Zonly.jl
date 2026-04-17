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
        "--T-Zgate"
            help = "Z gate duration"
            arg_type = Float64
            default = 72
        "--control-degree"
            help = "Degree of the splines defining control signals"
            arg_type = Int
            default = 4
        "--control-splines"
            help = "Number of basis functions for the splines defining control signals"
            arg_type = Int
            default = 12
        "--opt-iters"
            help = "Maximum number of iterations for IPOPT control signal optimization"
            arg_type = Int
            default = 100
        "--opt-dt"
            help = "Timestep size when performing the optimization"
            arg_type = Float64
            default = 0.05
        ##### MISC ####
        "--trials"
            help = "Number of independent trials to perform"
            arg_type = Int
            default = 2
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
    gate = PauliZ
    T_Zgate = parsed_args["T-Zgate"]
    
    # Controls
    degree    = parsed_args["control-degree"]
    n_splines = parsed_args["control-splines"]
    opt_dt    = parsed_args["opt-dt"]
    opt_iters = parsed_args["opt-iters"]

    n_trials = parsed_args["trials"] # number of trials to perform

    seed = parsed_args["seed"] # rng seed
    if seed < 1
        seed = Int(rand(UInt16))
    end

    # Number of times at which to evaluate the control functions
    # for plotting
    n_t_eval = 512

    ############################################################## 
    # SETUP
    ##############################################################

    # Set the seed for the RNG
    Random.seed!(seed)

    # Frequency parameter sampler
    omega_sampler = Normal(omega, omega_stdev)

    # Create a Qudit
    q = DigitalQudit(Ne, Ng)
    add_param_samples(q, omega, xi) # Initializes the param hist

    # Control for this gate
    local qcontrol = FortranBSplineControl(degree, n_splines, T_Zgate)
    add_control(q, PauliZ, qcontrol)
    # Infidelities for this gate
    q.infidelity[PauliZ] = History(Vector{Float64})


    ##############################################################
    # RUN EXPERIMENT
    ##############################################################

    controls = zeros(n_trials, 2, n_t_eval)
    control_coeffs = zeros(n_trials, 2*n_splines)
    omega_samples  = zeros(n_trials, n_samples)

    for t = 1:n_trials

        t_start = time()
        @printf("\n==== TRIAL %d of %d ==== \n", t, n_trials)

        # Parameter samples
        omega_samples[t,:] .= sort(rand(omega_sampler, n_samples))
        xi_samples = xi * ones(n_samples)
        update_param_samples(q, omega_samples[t,:], xi_samples)

        # Randomize the controls
        randomize_coeffs(q.controls[PauliZ])
        # Optimize the gate
        optimize_control(q, PauliZ, 
                            dt=opt_dt,
                            options=["max_iter" => opt_iters, "print_level" => 2, "limited_memory_max_history" => 250])
        # Plot the control signal in the rotating frame
        local control_obj = q.controls[PauliZ].objs.values[end]
        control_coeffs[t,:] = q.controls[PauliZ].coeffs.values[end]
        controls[t,1,:] = [
                eval_p_derivative(
                    control_obj, time, control_coeffs[t,:], 0
                )
                for time in LinRange(0, T_Zgate, n_t_eval)
            ]
        controls[t,2,:] = [
                eval_q_derivative(
                    control_obj, time, control_coeffs[t,:], 0
                )
                for time in LinRange(0, T_Zgate, n_t_eval)
            ]
        
        t_end = time()
        @printf("Done. Trial Runtime = %.2f min\n", 
                (t_end-t_start)/60)
        @printf("Program Elapsed Time = %.2f min\n", 
                (t_end-t_main_start)/60)
        
    end

    ##############################################################
    # SAVE DATA TO FILE
    ##############################################################

    output_dir = "./data/single_qudit_risk_neutral_variance"
    mkpath(output_dir)

    filename = @sprintf(
        "Zonly_N%d+%d_stdev%.1e_samples%d_ctrl%d-%d_dt%.2e_trials%d_seed%d.jld2",
        Ne, Ng, omega_stdev, n_samples, degree, n_splines, opt_dt, n_trials, seed
    )
    full_path = joinpath(output_dir, filename)

    jldsave(full_path; 
            Ne, Ng, omega, xi,  
            T_gate, degree, n_splines,  
            omega_stdev, n_samples, omega_samples, seed, 
            controls, control_coeffs,  
    )


    ##############################################################
    # PLOT CONTROL SIGNALS
    ##############################################################

    f = plot(
            title=L"Controls by Trial, $Z$ Gate", 
            titlefontsize=16, 
            xlabel=L"Time $t$", xlabelfontsize=14,
            ylabel="Control Amplitude", ylabelfontsize=14, 
            size=(600,500), 
            ylim=(-0.1,0.1),
            dpi=512,
            bottom_margin=10mm, top_margin=10mm,
            left_margin = 10mm, right_margin = 10mm
        )
    for t = 1:n_trials
        plot!(
            LinRange(0, T_Zgate, n_t_eval),
            controls[t,1,:],
            color=:blue, alpha=0.5, legend=(t==1),
            label="Real"
        )
        plot!(
            LinRange(0, T_Zgate, n_t_eval),
            controls[t,2,:],
            color=:red, alpha=0.5, legend=(t==1),
            label="Imag"
        )
    end

    # Create figure output directory
    fig_dir = "./figures/single_qudit_risk_neutral_variance"
    mkpath(fig_dir)

    # Saving to file
    filename = @sprintf(
        "Zonly_N%d+%d_stdev%.1e_samples%d_ctrl%d-%d_dt%.2e_trials%d_seed%d_controls.svg",
        Ne, Ng, omega_stdev, n_samples, degree, n_splines, opt_dt, n_trials, seed
    )

    fig_path = joinpath(fig_dir, filename)
    savefig(f, fig_path)

    return f

end # end main()


f = main()