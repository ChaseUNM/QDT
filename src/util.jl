# Used for plotting subscripts on labels
_subdigits = Dict('0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄','5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉')
function int_to_subscript(n::Integer)
    s = string(n)
    out = IOBuffer()
    for c in s
        print(out, Base.get(_subdigits, c, c))
    end
    return String(take!(out))
end