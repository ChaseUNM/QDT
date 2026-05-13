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
    delta = ω - ωr
    prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)
    event_obs = eval_forward(prob, control, pcof_optimal)
    event_obs = abs2.(event_obs)
    return event_obs 
end


