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


function update!(hist::History{I,V}, iter::I, value::V) where {I <: Real, V} 
    # Updates the value in the history associated with 
    # timestamp/iteration 'iter'. Set 'iter' to -1 to 
    # update to the latest timestamp.
    if iter == -1
        return hist.values[end] = value
    else
        idx = findfirst(hist.iterations .== iter)
        if idx === nothing
            throw("Iteration not found!")
        end
        hist.values[idx] = value
    end
end