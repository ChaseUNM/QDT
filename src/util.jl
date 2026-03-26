using ValueHistories

function get(hist::History{I,V}, iter::I=-1) where {I <: Real, V} 
    # Returns the value at iteration 'iter' of the history 
    # If iter = -1, returns the last set specifically
    if iter == -1
        return hist.values[end]
    else
        idx = findfirst(hist.iterations .== iter)
        if idx === nothing
            throw("Iteration not found!")
        end
        return hist.values[idx]
    end
end