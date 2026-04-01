using LinearAlgebra, QuantumGateDesign, ValueHistories
include("gates.jl")
include("controls.jl")
include("util.jl")
include("digital_qudit.jl")

################################################################
# Struct definition
################################################################

mutable struct DigitalQuditPair

    # FIELDS
    qudit1::DigitalQudit
    qudit2::DigitalQudit
    xi::History{Int64,Vector{Float64}}
    J::History{Int64,Vector{Float64}}
    controls::Dict{GateType, Vector{QuditControl}}
    infidelity::Dict{GateType, History{Int64, Float64}}
    
    # CONSTRUCTOR
    function DigitalQuditPair(q1::DigitalQudit, q2::DigitalQudit)
        xi = History(Vector{Float64})
        J  = History(Vector{Float64})
        controls = Dict{GateType, Vector{QuditControl}}()
        infidelity = Dict{GateType, History{Int64, Float64}}()
        new(q1, q2, xi, J, controls, infidelity)
    end
end




##################################################################
# SETTING/UPDATING PARAMETERS
##################################################################

function add_param_samples(
                    self::DigitalQuditPair, 
                    xi::Vector{Float64},
                    J::Vector{Float64};
                    iter::Int64=-1
    )
    # Adds a new sample of cross-Kerr and Jaynes-Cummings
    # couplings for this pair of qudits.
    #
    # By default, 'iter' is set to the latest iteration + 1
        
    # Verify the number of xi and J parameters match
    if length(xi) != length(J)
        throw("DigitalQuditPair::add_param_samples(): Number of xi and J samples must match!")
    end

    push!(self.xi, iter, xi)
    push!(self.J,  iter, J)
end

function add_param_samples(self::DigitalQuditPair, 
                           xi::Float64, J::Float64; 
                           kwargs...
    )
    # Overloaded version of the above function to work with 
    # xi and J as floats rather than vectors of floats
    add_param_samples(self, [xi], [J], kwargs...)
end



function update_param_samples(
                    self::DigitalQuditPair, 
                    xi::Vector{Float64};
                    J::Vector{Float64}
    )
    # Updates the sample cross-Kerr and Jaynes-Cummings coupling 
    # coefficients to this object's history at the latest 
    # timestamp in its history.
    
    # Throw error if the parameter histories are empty. 
    if length(self.xi) == 0 || length(self.J) == 0
        throw("Cannot update parameters of empty qudit history!")
    end

    # Verify the number of xi and J parameters match
    if length(xi) != length(J)
        throw("DigitalQuditPair::update_param_samples(): Number of xi and J samples must match!")
    end

    self.xi.values[end] = xi
    self.J.values[end] = J 
end



function add_control(self::DigitalQuditPair, 
                     gate::GateType, 
                     control_obj1::AbstractControl,
                     control_obj2::AbstractControl
                    )
    # Adds an entry to the control history for the provided gate.
    # If timestamp 'iter' isn't provided, the control is added to 
    # the history at 1 + {the latestest control timestamp}

    # Verify the gate times are the same
    if control_obj1.tf != control_obj1.tf
        throw("DigitalQuditPair::add_control(): control_obj1 and control_obj2 should have the same gate durations tf")
    end

    # Create a dictionary entry for this gate if 
    # this isn't one already
    if !haskey(self.controls, gate)
        self.controls[gate] = [
            QuditControl(control_obj1), 
            QuditControl(control_obj2)
        ]
        randomize_coeffs(self.controls[gate][1])
        randomize_coeffs(self.controls[gate][2])
    else
        throw("DigitalQuditPair::add_control(): Not yet implemented!")
    end
end


##################################################################
# SIMULATION ROUTINES
##################################################################


function get_drift_hamiltonians(self::DigitalQuditPair)
    # Returns a Vector of Matrices representing the drift
    # Hamiltonian (in the rotating frames) for each of this 
    # pair of qudits current parameter samples

    q1 = self.qudit1
    q2 = self.qudit2
    subsystem_sizes = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]

    # Lowering operator by subsystem
    a1 = promote_subsys_op(lower_op(subsystem_sizes[1]), subsystem_sizes, 1)
    a2 = promote_subsys_op(lower_op(subsystem_sizes[2]), subsystem_sizes, 2)

    # Drift Hamiltonians by subsystem
    H_drift_1 = get_drift_hamiltonians(q1)
    H_drift_2 = get_drift_hamiltonians(q2)

    # Coupling parameters
    _, xi = last(self.xi)
    _, J = last(self.J)
    n_samples = length(xi)    

    # Verify the number of parameters for the subsystems
    # matches with the number of coupling parameters
    if n_samples != num_samples(self.qudit1) || n_samples != num_samples(self.qudit2)
        throw("DigitalQuditPair::get_drift_hamiltonians(): Number of coupling parameters doesnt match individual qudit parameters")
    end

    # Full drift Hamiltonians
    N = prod(subsystem_sizes)
    H_drift = Array{Float64}(undef, n_samples, N, N)
    for j = 1:n_samples
        H_drift[j,:,:] = (
            promote_subsys_op(H_drift_1[j], subsystem_sizes, 1) 
            + promote_subsys_op(H_drift_2[j], subsystem_sizes, 2) 
            - xi[j] * (a1'*a1)*(a2'*a2)
            + J[j] * (a1 * a2' + a1' * a2)
        )
    end
    return H_drift
end


function get_control_hamiltonians(self::DigitalQuditPair)
    # Returns a Vector of Matrices representing the real and 
    # imaginary parts of the control Hamiltonians for each of 
    # this pair of qudits current parameter samples.
    # These control Hamiltonians are of the same size as the 
    # *full* system

    # Subsystem sizes
    q1 = self.qudit1
    q2 = self.qudit2
    n = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]

    # Control Hamiltonians of the subsystems
    H_c_re_1, H_c_im_1 = get_control_hamiltonians(q1)
    H_c_re_2, H_c_im_2 = get_control_hamiltonians(q2)

    # Promote to the full system
    H_c_re = [
        promote_subsys_op(H_c_re_1, n, 1),
        promote_subsys_op(H_c_re_2, n, 2),  
    ]
    H_c_im = [
        promote_subsys_op(H_c_im_1, n, 1),
        promote_subsys_op(H_c_im_2, n, 2),  
    ]

    return H_c_re, H_c_im

end



function get_schrodinger_problems(self::DigitalQuditPair, T, dt)
    # Returns a Vector of SchrodingerProb objects, one for each 
    # of this pair of qudit's current parameter samples.
    # Argument 'T' specifies the time integration interval [0,T]
    # and dt is the stepsize
    #

    q1 = self.qudit1
    q2 = self.qudit2

    # Initial state
    n = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]
    n_ess = [q1.Ne, q2.Ne]
    U0 = initial_states(n, n_ess)

    # Drift Hamiltonians for the full system
    H_drift = get_drift_hamiltonians(self)

    # Unscaled control Hamiltonian by subsystem
    H_c_re, H_c_im = get_control_hamiltonians(self)

    # Number of timesteps
    n_timesteps = ceil(Int, T/dt)

    # Generating the SchrodingerProb
    probs = [  SchrodingerProb(
                    H_drift[j,:,:], 
                    H_c_re, H_c_im, 
                    U0, T, n_timesteps
                ) 
                for j in 1:size(H_drift,1)
            ]
    
    return probs
end



function run_control(
        self::DigitalQuditPair, 
        q1_control::QuditControl,
        q2_control::QuditControl; 
        dt=0.2
    )
    # Computes the terminal state Psi, for each of the qudit's current
    # parameter settings, in response to the provided control signal

    # Extract control variables
    q1_control = self.controls[gate][1]
    q2_control = self.controls[gate][2]
    _, control_obj1 = last(q1_control.objs)
    _, control_obj2 = last(q2_control.objs)
    T_gate = control_obj1.tf
    _, control_coeffs1 = last(q1_control.coeffs)
    _, control_coeffs2 = last(q2_control.coeffs)
    
    # Create a SchrodingerProb for each of the qudits param samples
    probs = get_schrodinger_problems(self, T_gate, dt)  
    n_probs = length(probs)  
    (N, Ne) = size(probs[1].u0)

    # Run simulation for each parameter setting    
    Psi = zeros(Complex, n_probs, N, Ne)
    for j = 1:n_probs
        state_history = eval_forward(
                            probs[j], 
                            [control_obj1, control_obj2], [control_coeffs1; control_coeffs2]
                        )
        Psi[j,:,:] = state_history[:,end,:]
    end

    return Psi
end



function optimize_control(
            self::DigitalQuditPair, 
            gate::GateType; 
            dt = 0.2,
            options=["max_iter" => 100, "print_level" => 3] 
    )
    # Optimizes the control signals for this qudit to implement 
    # the provided 'gate'

    q1 = self.qudit1
    q2 = self.qudit2

    # Target unitary: CNOT with qudit1 as the control qudit
    n = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]
    n_ess = [q1.Ne, q2.Ne]
    U_target = unitary(gate, [1,2], n, n_ess)

    # Extract control variables
    q1_control = self.controls[gate][1]
    q2_control = self.controls[gate][2]
    max_amplitude = q1_control.max_amplitude
    iter, control_obj1 = last(q1_control.objs)
       _, control_obj2 = last(q2_control.objs)
    T_gate = control_obj1.tf
    _, control_coeffs1 = last(q1_control.coeffs)
    _, control_coeffs2 = last(q2_control.coeffs)
    
    # Create a SchrodingerProb for each of the qudits param samples
    probs = get_schrodinger_problems(self, T_gate, dt)    

    # Run the optimizer
    opt_ret_multiple = optimize_prob(
                            probs, [control_obj1, control_obj2], [control_coeffs1; control_coeffs2], U_target, 
                            pcof_lbound=-max_amplitude, pcof_ubound=max_amplitude, cost_type=:Infidelity, ipopt_options=options
                        )

    # Save Infidelity     
    push!(self.infidelity[gate], iter, opt_ret_multiple.obj_val)
    # Save updated spline coefficients
    n_coeffs1 = length(control_coeffs1)
    control_coeffs1 .= opt_ret_multiple.x[1:n_coeffs1]
    control_coeffs2 .= opt_ret_multiple.x[n_coeffs1+1:end]

end

