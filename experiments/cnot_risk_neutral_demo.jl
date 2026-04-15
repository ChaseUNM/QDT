using ArgParse, LinearAlgebra, Random, Distributions, Printf
using QuantumGateDesign
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
using JLD2
include("../src/digital_qudit.jl")
include("../src/digital_device.jl")

################################################################
# SETUP ARGUMENT PARSER
################################################################

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
            default = 0
        "--omega1"
            help = "Qudit 1 true frequency"
            arg_type = Float64
            default = 4.5
        "--omega2"
            help = "Qudit 2 true frequency"
            arg_type = Float64
            default = 4.8
        "--xi1"
            help = "Qudit 1 true self-kerr"
            arg_type = Float64
            default = 0.21
        "--xi2"
            help = "Qudit 2 true self-kerr"
            arg_type = Float64
            default = 0.23
        "--xi12"
            help = "True cross-kerr coupling between the qudits"
            arg_type = Float64
            default = 0.1
        "--omega-stdev"
            help = "Sampling stdev for qudit frequency"
            arg_type = Float64
            default = 1E-3
        "--xi-self-stdev"
            help = "Sampling stdev for self-kerr coeffs"
            arg_type = Float64
            default = 1E-3
        "--xi-cross-stdev"
            help = "Sampling stdev for cross-kerr coeffs"
            arg_type = Float64
            default = 1E-3
        "--samples"
            help = "Number of parameter samples"
            arg_type = Int
            default = 10
        "--samples-eval"
            help = "Number of parameter samples used for control evaluation"
            arg_type = Int
            default = 500
        ##### GATES ####
        "--T-gate"
            help = "Gate duration"
            arg_type = Float64
            default = 550
        "--control-degree"
            help = "Degree of the splines defining control signals"
            arg_type = Int
            default = 2
        "--control-splines"
            help = "Number of basis functions for the splines defining control signals"
            arg_type = Int
            default = 8
        "--control-harmonics"
            help = "Number of carrier frequencies 0, xi12, 2*xi, ... in the control signals"
            arg_type = Int
            default = 2
        "--warmup-iters"
            help = "Maximum number of iterations for IPOPT control signal optimization during the control warmup"
            arg_type = Int
            default = 32
        "--warmup-samples"
            help = "Number of samples to use during the control warmup"
            arg_type = Int
            default = 3
        "--opt-iters"
            help = "Maximum number of iterations for IPOPT control signal optimization"
            arg_type = Int
            default = 100
        ##### MISC ####
        "--seed"
            help = "Seed for the global RNG"
            arg_type = Int
            default = -1 # Selected randomly
    end

    return parse_args(s)
end

################################################################
# Function to convert control coeffs to signals
################################################################
function control_vs_time(control_obj, coeffs, T, n_steps)
    control_re = [
        eval_p_derivative(control_obj, time, coeffs, 0)
        for time in LinRange(0, T, n_steps)
    ]
    control_im = [
        eval_q_derivative(control_obj, time, coeffs, 0)
        for time in LinRange(0, T, n_steps)
    ]
    return [control_re control_im]
end


################################################################
# Main Method
################################################################
function main()

    ############################################################
    # PARSE ARGUMENTS
    ############################################################

    parsed_args = parse_commandline()

    # Physical Device 
    Ne       = parsed_args["Ne"]
    Ng       = parsed_args["Ng"]
    omega1   = parsed_args["omega1"]
    omega2   = parsed_args["omega2"]
    xi1      = parsed_args["xi1"]
    xi2      = parsed_args["xi2"] 
    xi12     = parsed_args["xi12"] 
               # Artificially large to allow fast coupling. 
               # Actual value: 1e-6 
    J12    = 0.0  # REMOVED DUE TO OMISSION IN IMPLEMENTATION!

    # Parameter Uncertainies
    omega_stdev     = parsed_args["omega-stdev"] 
    xi_self_stdev   = parsed_args["xi-self-stdev"]
    xi_cross_stdev  = parsed_args["xi-cross-stdev"]
    J12_stdev       = 0
    n_samples       = parsed_args["samples"]
    n_samples_eval  = parsed_args["samples-eval"]
    seed = parsed_args["seed"] # rng seed
    if seed < 1
        seed = Int(rand(UInt16))
        parsed_args["seed"] = seed
    end

    # Controls
    gate = CNOT
    T_gate      = parsed_args["T-gate"]
    degree      = parsed_args["control-degree"]
    n_splines   = parsed_args["control-splines"]
    n_harmonics = parsed_args["control-harmonics"]
    opt_iters   = parsed_args["opt-iters"]

    warmup_iters   = parsed_args["warmup-iters"]
    warmup_samples = parsed_args["warmup-samples"]
    warmup_samples = min(warmup_samples, n_samples)


    ############################################################
    # REDIRECT STDOUT
    ############################################################

    output_dir = "./data/cnot_risk_neutral_demo"
    mkpath(output_dir)

    filename_prefix = @sprintf(
        "N%d+%d_stdev%.1e_samples%d_harmonics%d_seed%d_warmup%dx%d_iters%d",
        Ne, Ng, omega_stdev, n_samples, n_harmonics, 
        seed, warmup_samples, warmup_iters, opt_iters
    )

    filename = joinpath(output_dir, filename_prefix*".out")
    output_file = open(filename, "w")
    redirect_stdout(output_file)

    # Write experiment config to file
    for (key, value) in parsed_args
        @printf("%s => %s\n", key, string(value))
    end
    flush(stdout)

    ############################################################
    # SETUP
    ############################################################

    # Set the seed for the RNG
    Random.seed!(seed)

    # Create parameter samplers
    omega1_sampler = Normal(omega1, omega_stdev)
    omega2_sampler = Normal(omega2, omega_stdev)
    xi1_sampler   = Normal(xi1,  xi_self_stdev)
    xi2_sampler   = Normal(xi2,  xi_self_stdev)
    xi12_sampler  = Normal(xi12, xi_cross_stdev)
    J12_sampler   = Normal(J12,  J12_stdev)

    # Parameter samples coming from "characterization"
    omega1_samples = rand(omega1_sampler, n_samples)
    omega2_samples = rand(omega2_sampler, n_samples)
    xi1_samples   = rand(xi1_sampler, n_samples)
    xi2_samples   = rand(xi2_sampler, n_samples)
    xi12_samples  = rand(xi12_sampler, n_samples)
    J12_samples   = rand(J12_sampler, n_samples)
    opt_samples = Dict(
        "omega1" => copy(omega1_samples),
        "omega2" => copy(omega2_samples),
        "xi1"    => copy(xi1_samples),
        "xi2"    => copy(xi2_samples),
        "xi12"   => copy(xi12_samples)
    )

    # Create two digital qudits
    q1 = DigitalQudit(Ne, Ng)
    q2 = DigitalQudit(Ne, Ng)

    # Put the qubits into a pair 
    pair = DigitalQuditPair(q1, q2)

    # Create the controls for the CNOT gate
    base_control = FortranBSplineControl(degree, n_splines, T_gate)
    xi12_mean = mean(xi12_samples)
    carrier_freqs = [-xi12_mean*i for i = 0:n_harmonics]
    q1_control = CarrierControl(base_control, carrier_freqs)
    q2_control = CarrierControl(base_control, carrier_freqs)
    add_control(pair, CNOT, q1_control, q2_control)
    # Infidelity for this gate
    pair.infidelity[CNOT] = History(Float64)


    ################################################################
    # WARM-UP
    #
    # Optimize the controls at the parameter samples closest to the sample mean
    ################################################################

    @printf("\n\n======== CONTROL WARM-UP ========\n")
    flush(stdout)

    t_warmup_start = time()

    # Set the digital qudits parameters to sample closest to the sample mean in terms of frequencies. 
    # These parameters seem to be the most important in terms of control optimization.
    #
    # Selecting an index
    omega_samples = [omega1_samples omega2_samples];
    omega_mean = [mean(omega1_samples) mean(omega2_samples)]
    dist_to_mean = norm.(eachrow(omega_samples .- omega_mean))
    order = sortperm(dist_to_mean)
    i1 = order[1:warmup_samples]
    #
    # Setting the parameters
    add_param_samples(q1, omega1_samples[i1], xi1_samples[i1])
    add_param_samples(q2, omega2_samples[i1], xi2_samples[i1])
    add_param_samples(pair, xi12_samples[i1], J12_samples[i1])
    #
    # Set the rotating frequencies of the qubits to the sample means
    q1.omega_rot = omega_mean[1]
    q2.omega_rot = omega_mean[2]

    # Optimize the control for this point estimate of the parameters
    optimize_control(pair, CNOT, options=[
                                    "max_iter" => warmup_iters, 
                                    "print_level" => 5, "limited_memory_max_history" => 250
                                ])

    # Set the digital device parameters to store the samples
    update_param_samples(q1, omega1_samples, xi1_samples)
    update_param_samples(q2, omega2_samples, xi2_samples)
    update_param_samples(pair, xi12_samples, J12_samples) 

    # Save the control coeffs at this point
    control_coeffs1_post_warmup = pair.controls[CNOT][1].coeffs.values[end]
    control_coeffs2_post_warmup = pair.controls[CNOT][2].coeffs.values[end]
    control_coeffs_post_warmup = [control_coeffs1_post_warmup control_coeffs2_post_warmup]

    # Compute the control signals as functions of time
    n_t_eval = Int(T_gate / 0.1)
    control_obj = pair.controls[CNOT][1].objs.values[end]
    controls1_post_warmup = control_vs_time(
                                control_obj, 
                                control_coeffs1_post_warmup,
                                T_gate, n_t_eval
                            )
    control_obj = pair.controls[CNOT][2].objs.values[end]
    controls2_post_warmup = control_vs_time(
                                control_obj, 
                                control_coeffs2_post_warmup,
                                T_gate, n_t_eval
                            )

    # Fidelity for each parameter sample
    dt = 0.05
    Psi = run_control(pair, 
                    pair.controls[CNOT][1], 
                    pair.controls[CNOT][2], 
                    dt=dt)
    infidelities_post_warmup = zeros(n_samples)
    U = unitary(CNOT, Ne*Ne)
    for j = 1:n_samples
        Psi_j = Psi[j,:,:]
        foreach(normalize!, eachcol(Psi_j))
        infidelities_post_warmup[j] = infidelity(Psi_j, U, Ne*Ne)
    end

    @printf("\nWarm-up Runtime: %.2f s\n", time()-t_warmup_start)
    flush(stdout)

    ################################################################# RISK NEUTRAL OPTIMIZATION
    ################################################################

    t_optim_start = time()

    @printf("\n\n======== RISK NEUTRAL CONTROL ========\n")
    flush(stdout)

    # Set the digital device parameters to store the samples
    update_param_samples(q1, omega1_samples, xi1_samples)
    update_param_samples(q2, omega2_samples, xi2_samples)
    update_param_samples(pair, xi12_samples, J12_samples) 

    # Optimize the controls in the risk neutral setting.
    # This will start from the control signals determined 
    # during the warm-up period
    optimize_control(pair, CNOT, 
                    options=[
                        "max_iter" => opt_iters, 
                        "print_level" => 5, "limited_memory_max_history" => 250
                        ]
                    )

    # Save the final control coeffs 
    control_coeffs1 = pair.controls[CNOT][1].coeffs.values[end]
    control_coeffs2 = pair.controls[CNOT][2].coeffs.values[end]
    control_coeffs = [control_coeffs1 control_coeffs2]

    # Compute the control signals as functions of time
    control_obj = pair.controls[CNOT][1].objs.values[end]
    controls1 = control_vs_time(
                    control_obj, 
                    control_coeffs1,
                    T_gate, n_t_eval
                )
    control_obj = pair.controls[CNOT][2].objs.values[end]
    controls2 = control_vs_time(
                    control_obj, 
                    control_coeffs2,
                    T_gate, n_t_eval
                )

    @printf("\nOptimization Runtime: %.2f s\n", time()-t_optim_start)
    flush(stdout)

    ############################################################
    # EVAL CONTROLS ON LARGER SAMPLE SET
    ############################################################
    
    @printf("\n\n======== CONTROL EVALUATION ========\n")
    flush(stdout)

    # Generate a larger sample set
    omega1_samples = rand(omega1_sampler, n_samples_eval)
    omega2_samples = rand(omega2_sampler, n_samples_eval)
    xi1_samples    = rand(xi1_sampler,    n_samples_eval)
    xi2_samples    = rand(xi2_sampler,    n_samples_eval)
    xi12_samples   = rand(xi12_sampler,   n_samples_eval)
    J12_samples    = rand(J12_sampler,    n_samples_eval)
    eval_samples = Dict(
        "omega1" => omega1_samples,
        "omega2" => omega2_samples,
        "xi1"    => xi1_samples,
        "xi2"    => xi2_samples,
        "xi12"   => xi12_samples
    )

    # Add this sample set to the digit QC components
    update_param_samples(q1, omega1_samples, xi1_samples)
    update_param_samples(q2, omega2_samples, xi2_samples)
    update_param_samples(pair, xi12_samples, J12_samples) 

    # Set the rotating frequencies of the qubits to the means of the original samples
    q1.omega_rot = omega_mean[1]
    q2.omega_rot = omega_mean[2]

    # Evaluating the controls
    # Fidelity for each parameter sample
    dt = 0.01
    Psi = run_control(pair, 
                    pair.controls[CNOT][1], 
                    pair.controls[CNOT][2], 
                    dt=dt)
    infidelities = zeros(n_samples_eval)
    U = unitary(CNOT, Ne+Ng)
    for j = 1:n_samples_eval
        Psi_j = Psi[j,:,:]
        foreach(normalize!, eachcol(Psi_j))
        infidelities[j] = infidelity(Psi_j, U, Ne*Ne)
        if mod(j,10) == 0
            @printf("... %d of %d samples done\n", j, n_samples_eval)
            flush(stdout)
        end
    end


    ############################################################
    # SAVE DATA TO FILE
    ############################################################

    data_path = joinpath(output_dir, filename_prefix*".jld2")

    jldsave(data_path; 
            Ne, Ng, 
            omega1, omega2, xi1, xi2, xi12,  
            omega_stdev, xi_self_stdev, xi_cross_stdev, n_samples, seed, opt_samples, 
            T_gate, degree, n_splines, carrier_freqs, 
            control_coeffs_post_warmup, 
            controls1_post_warmup, controls2_post_warmup,
            control_coeffs, 
            controls1, controls2,
            infidelities_post_warmup, 
            eval_samples, infidelities
    )

    # Close output file --- not needed any more
    close(output_file)

end # end main()


main()