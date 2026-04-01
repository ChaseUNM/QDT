using LinearAlgebra, QuantumGateDesign, ValueHistories


mutable struct QuditControl

    ### FIELDS
    max_amplitude::Float64
    objs::History{Int64, AbstractControl}
    coeffs::History{Int64, Vector{Float64}}
    
    ### CONSTRUCTORS

    function QuditControl(;max_amplitude=0.1)
        objs = History(AbstractControl)
        coeffs = History(Vector{Float64})
        new(max_amplitude, objs, coeffs)
    end

    function QuditControl(c::AbstractControl; iter=0, max_amplitude=0.1)
        objs = History(AbstractControl)
        push!(objs, c)
        coeffs = History(Vector{Float64})
        new(max_amplitude, objs, coeffs)
    end

end   


function lastiter(c::QuditControl)
    # Returns the timestamp of the last control logged in 
    # this object's history
    return max(c.objs.lastiter, 0) 
end



function randomize_coeffs(c::QuditControl)
    # Sets the spline coefficients of this control the values
    # drawn from U[-0.5,0.5] * max_amplitude

    # Verify the control already has an entry that we can
    # build off of
    if length(c.coeffs) == 0 && length(c.objs) == 0
        throw("Empty QubitControl cannot be randomized")
    end

    iter, control_obj = last(c.objs)
    N_coeff = control_obj.N_coeff

    # Generate the first set of random coefficients
    if c.coeffs.lastiter < iter
        coefs = (0.5 .- rand(N_coeff)) .* c.max_amplitude
        push!(c.coeffs, iter, coefs)

    # Randomize existing coefficients
    else 
        coefs = c.coeffs.values[end]
        coefs .= (0.5 .- rand(N_coeff)) .* c.max_amplitude
    end
    
end



function randomize(iter::Int64, c::QuditControl)
    # Sets the spline coefficients of this control the values
    # drawn from U[-0.5,0.5] * max_amplitude

    # Verify the control already has an entry that we can
    # build off of
    if length(c.coeffs) == 0
        throw("Empty QubitControl cannot be initialized")
    end

    # Pointer to the previous control object
    last_obj = last(c.objs)
    push!(c, iter, last_obj)

    # Randomize the coefficients
    N_coeff = size(last(c.spline_coeffs))
    new_coeffs = (0.5 .- rand(N_coeff)) .* c.max_amplitude
    push!(c.coeffs, iter, new_coeffs) 

end


