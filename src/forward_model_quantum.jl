############################
# src/forward_model_quantum.jl
############################

using QuantumGateDesign

# Evolve forward with pre-calculated control 

function forward_event_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal)
    H_control_real = [0.0 1;
    1 0]
    H_control_imag = [0.0 1;
    -1 0]
    real_control_ops = [H_control_real]
    imag_control_ops = [H_control_imag]
    control = FortranBSplineControl(degree, n_splines, T)
    # println("Evaluating forward model with control coefficients: ", pcof_optimal)
    # println("Degree: ", degree, " Number of splines: ", n_splines)
    delta = ω - ωr
    prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)
    event_obs = eval_forward(prob, control, pcof_optimal)
    event_obs = abs2.(event_obs)
    # normalize evolution so p_1 + p_2 = 1
    n_tot, _, n_ess = size(event_obs)
    for i in 1:nsteps + 1
        for j in 1:n_tot
        event_obs[:,i,j] = event_obs[:,i,j]/sum(event_obs[:,i,j])
        end 
    end
    return event_obs 
end

function forward_event_quantum_multi(Ne::Vector{<:Real}, Ng::Vector{<:Real}, ω::Vector{<:Real}, ωr::Vector{<:Real}, ξ::Vector{<:Real}, dipole::Real, cross_kerr::Real, degree::Real, n_splines::Real, U0::AbstractMatrix, T::Real, nsteps::Real, pcof_optimal::AbstractVector, carrier_freqs::Vector{<:Real})
    subsystem_sizes = [Ne[1]+Ng[1], Ne[2]+Ng[2]]
    N = prod(subsystem_sizes)
    
    a1 = promote_subsys_op(lower_op(subsystem_sizes[1]), subsystem_sizes, 1)
    a2 = promote_subsys_op(lower_op(subsystem_sizes[2]), subsystem_sizes, 2)
    H_drift_1 = get_drift_hamiltonians(q1)
    H_drift_2 = get_drift_hamiltonians(q2)
        # Set unscaled drift Hamiltonian
    a1 = lower_op(subsystem_sizes[1]) 
    a2 = lower_op(subsystem_sizes[2])
    H_omega_1 = a1' * a1
    H_omega_2 = a2' * a2 
    H_xi_1 = a1' * a1' * a1 * a1
    H_xi_2 = a2' * a2' * a2 * a2
    # Scaled Hamiltonians
    H_drift_1 =
        ((ω[1]-ωr[1])*H_omega_1) .- (0.5*ξ[1]*H_xi_1)

    H_drift_2 = 
        ((ω[2]-ωr[2])*H_omega_2) .- (0.5*ξ[2]*H_xi_2)
    a1_full = promote_subsys_op(lower_op(subsystem_sizes[1]), subsystem_sizes, 1)
    a2_full = promote_subsys_op(lower_op(subsystem_sizes[2]), subsystem_sizes, 2)
    
    H_drift_1_full = promote_subsys_op(H_drift_1, subsystem_sizes, 1)
    H_drift_2_full = promote_subsys_op(H_drift_2, subsystem_sizes, 2)
    cross_kerr_full = cross_kerr * (a1_full'*a1_full)*(a2_full'*a2_full)
    dipole_full = dipole * (a1_full * a2_full' + a1_full' * a2_full)
    H_drift = H_drift_1_full + H_drift_2_full - cross_kerr_full - dipole_full

    H_c_re_1 = a1 + a1'
    H_c_im_1 = a1 - a1'
    H_c_re_2 = a2 + a2'
    H_c_im_2 = a2 - a2'
    H_c_re = [
        promote_subsys_op(H_c_re_1, subsystem_sizes, 1),
        promote_subsys_op(H_c_re_2, subsystem_sizes, 2),  
    ]
    H_c_im = [
        promote_subsys_op(H_c_im_1, subsystem_sizes, 1),
        promote_subsys_op(H_c_im_2, subsystem_sizes, 2),  
    ]
    base_control1 = FortranBSplineControl(degree, n_splines, T)
    base_control2 = FortranBSplineControl(degree, n_splines, T)

    control1 = CarrierControl(
                base_control1, 
                (ω[1]-ωr[1]) .- carrier_freqs
             )
    control2 = CarrierControl(
                base_control2, 
                (ω[2]-ωr[2]) .- carrier_freqs
             )
    
    
    prob = SchrodingerProb(H_drift, H_c_re, H_c_im, U0, T, nsteps)
    event_obs = eval_forward(
            prob,
            [control1, control2],
            [pcof_optimal[1]; pcof_optimal[2]]
        )
    event_obs = abs2.(event_obs)
    for i in 1:nsteps + 1
        for j in 1:N
            event_obs[:,i,j] = event_obs[:,i,j]/sum(event_obs[:,i,j])
        end 
    end
    return event_obs
end