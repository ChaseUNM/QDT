using LinearAlgebra, QuantumGateDesign

##############################################################
# GateType enum
##############################################################
@enum GateType PauliX PauliY PauliZ Hadamard CNOT

SINGLE_QUDIT_GATES = [PauliX, PauliY, PauliZ, Hadamard]

##############################################################
# Functions to get unitaries
##############################################################

function unitary(gate::GateType)
    # Returns the unitary associated with the gate
    # as a 2x2 or 4x4 matrix
    if gate == PauliX
        return PauliX_gate()
    elseif gate == PauliY
        return PauliY_gate()
    elseif gate == PauliZ
        return PauliZ_gate()
    elseif gate == Hadamard
        return Hadamard_gate()
    elseif gate == CNOT
        return CNOT_gate()
    else
        throw("GateType::unitary() You shouldnt be here?")
    end
end


function unitary(gate::GateType, n::Int64)
    # Returns the unitary associated with the gate as an n by n_gate matrix
    Ug = unitary(gate)
    n_gate = size(Ug,1)
    if n_gate < n
        U = zeros(eltype(Ug), n, n_gate)
        U[1:n_gate,1:n_gate] = Ug
    else
        U = Ug
    end
    return U
end



function unitary(
            gate::GateType, 
            which_qudits::Vector{Int}, 
            n::Vector{Int},
            n_ess::Vector{Int}
    )
    # Arguments
    #
    #   gate            GateType, specifying which gate whose 
    #                   unitary to return
    #
    #   which_qudits    Vector of indices specifying which qudits
    #                   to apply the gate to 
    #
    #   n               Vector of subsystem sizes (essential + 
    #                   guard)
    #   
    #   n_ess           Vector of essential level by subsystem
    #
    
    # Single Qudit Gates
    if gate in SINGLE_QUDIT_GATES

        @assert length(which_qudits) == 1 "Single qudit gates should only specify one active qudit"
        
        i = which_qudits[1]
        @assert n_ess[i] == 2 "N_ess > 2 not yet implemented"

        # Identity operators for each subsystem. These exclude the columns for which guard levels have non-zero coefficients
        d = length(n)
        op = [Matrix{Float64}(I, n[k], n_ess[k]) for k = 1:d]

        # Replace index i with the unitary acting on that 
        # subsystem
        op[i] = unitary(gate, n[i])

        # Build operator as kronecker product of smaller matrices 
        return reduce(kron, op)

    # Two Qudit Gates
    elseif gate == CNOT
        
        # Note: This is an an inefficient implementation, which # # builds the full N x N CNOT(i,j) matrix and then applies 
        # the matrix to all initial states |i_1, ..., i_d> of with
        # i_k <= n_ess[k].

        @assert length(which_qudits) == 2 "CNOT gate should only specify two active qudits"

        i = which_qudits[1]
        j = which_qudits[2]

        # Full identity operators for each subsystem
        d = length(n)
        op = [Matrix{Float64}(I, n[k], n[k]) for k = 1:d]

        # Full CNOT(i,j) operator as N x N matrix
        # CNOT(i,j) = ... ⊗ |0><0| ⊗ ... ⊗ Id ⊗ ... +
        #             ... ⊗ |1><1| ⊗ ... ⊗  X ⊗ ... + 
        #             ... ⊗ |2><2| ⊗ ... ⊗ Id ⊗ ... +
        #             ...
        #             ... ⊗ |n><n| ⊗ ... ⊗ Id ⊗ ...
        #           = Id_N + 
        #             ... ⊗ |1><1| ⊗ ... ⊗ (X-Id) ⊗ ... 
        #
        #
        # Start with full N x N identity matrix
        CNOT_full = reduce(kron, op)
        # Add in ... ⊗ |1><1| ⊗ ... ⊗ (X-Id) ⊗ ...
        op[i] .= 0
        op[i][2,2] = 1
        op[j][1:2,1:2] = [-1 1 ; 1 -1];
        CNOT_full .+= reduce(kron, op)
        
        # Apply to all states psi = |i_1, ..., i_d> for which 
        # for which i_k <= n_ess[k]
        return CNOT_full * initial_states(n, n_ess)


    # Not yet implemented!
    else 
        throw("gates.jl::unitary(): requested gate not yet implemented!")
    end

end


##############################################################
# Initial States
##############################################################

function initial_states(n::Vector{Int}, n_ess::Vector{Int})
    # Returns the columns of the prod(n) x prod(n) identity 
    # matrix corresponding to states psi = |i_1, ..., i_d> of a
    # composite system for which i_k <= n_ess[k]

    d = length(n)

    # Identity operators for each subsystem. These exclude the columns for which guard levels have non-zero coefficients
    Id = [Matrix{Float64}(I, n[k], n_ess[k]) for k = 1:d]

    # Id[1] ⊗ Id[2] ⊗ ... ⊗ Id[d] 
    return reduce(kron, Id)
end