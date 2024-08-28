using LinearAlgebra
using Random
using Printf

const BLOCK_DEPTH = 2
const BLOCK_SIZE = 1 << BLOCK_DEPTH

struct Block
    data::Matrix{Float64}
end
#define _00 0
#define _01 1
#define _10 2
#define _11 3

#define TR_00 _00
#define TR_01 _10
#define TR_10 _01
#define TR_11 _11
const TOP_LEFT = 1
const TOP_RIGHT = 2
const BOTTOM_LEFT = 3
const BOTTOM_RIGHT = 4
const TRANS_TOP_LEFT = 1
const TRANS_TOP_RIGHT = 3
const TRANS_BOTTOM_LEFT = 2
const TRANS_BOTTOM_RIGHT = 4

Block() = Block(zeros(BLOCK_SIZE, BLOCK_SIZE))

abstract type Node end

struct LeafNode <: Node
    block::Block
end

struct InternalNode <: Node
    children::Vector{Union{Node,Nothing}} # TODO: maybe use a tuple instead
end

InternalNode() = InternalNode([nothing, nothing, nothing, nothing])

function block_schur_full!(B::Block, A::Block, C::Block)
    for i in 1:BLOCK_SIZE
        for j in 1:BLOCK_SIZE
            for k in 1:BLOCK_SIZE
                B.data[i, j] -= A.data[i, k] * C.data[j, k]
            end
        end
    end
end

function block_schur_half!(B::Block, A::Block, C::Block)
    for i in 1:BLOCK_SIZE
        for j in 1:i
            for k in 1:BLOCK_SIZE
                B.data[i, j] -= A.data[i, k] * C.data[j, k]
            end
        end
    end
end
function block_backsub!(B, L)
    n = size(B, 1)
    for j in 1:n
        for i in 1:n
            for k in 1:(i-1)
                B[i, j] -= L[i, k] * B[k, j]
            end
        B[i, j] /= L[i, i]
        end
    end
end
function block_backsub!(B::Block, L::Block)
    for i in 1:BLOCK_SIZE
        for j in 1:BLOCK_SIZE
            for k in 1:(i-1)
                B.data[i, j] -= L.data[i, k] * L.data[k, j]
            end
        B.data[i, j] /= L.data[i, i]
        end
    end
end

function block_cholesky!(B::Block)
    for k in 1:BLOCK_SIZE
        if B.data[k, k] < 0
            @show B.data
            error("Matrix is not numerically stable")
        end
        x = sqrt(B.data[k, k])
        for i in k:BLOCK_SIZE
            B.data[i, k] /= x
        end
        for j in (k+1):BLOCK_SIZE
            for i in j:BLOCK_SIZE
                B.data[i, j] -= B.data[i, k] * B.data[j, k]
                if (j > i && B.data[j, i] != 0)
                    @show (i, j, B.data[j, j], "Upper not")
                end
            end
        end
    end
end

function new_block_leaf()
    LeafNode(Block())
end

function new_internal(a00, a01, a10, a11)
    InternalNode([a00, a01, a10, a11])
end

function copy_matrix(depth::Int, a::Union{Node,Nothing})
    if isnothing(a)
        return nothing
    end

    if depth == BLOCK_DEPTH
        return LeafNode(Block(copy(a.block.data)))
    end

    depth -= 1
    r00 = Threads.@spawn copy_matrix(depth, a.children[TOP_LEFT])
    r01 = Threads.@spawn copy_matrix(depth, a.children[TOP_RIGHT])
    r10 = Threads.@spawn copy_matrix(depth, a.children[BOTTOM_LEFT])
    r11 = copy_matrix(depth, a.children[BOTTOM_RIGHT])

    return new_internal(fetch(r00), fetch(r01), fetch(r10), r11)
end

function get_matrix(depth::Int, a::Union{Node,Nothing}, r::Int, c::Int)
    if isnothing(a)
        return 0.0
    end

    if depth == BLOCK_DEPTH
        return a.block.data[r+1, c+1]
    else
        depth -= 1
        mid = 1 << depth
        if r < mid
            if c < mid
                return get_matrix(depth, a.children[TOP_LEFT], r, c)
            else
                return get_matrix(depth, a.children[TOP_RIGHT], r, c - mid)
            end
        else
            if c < mid
                return get_matrix(depth, a.children[BOTTOM_LEFT], r - mid, c)
            else
                return get_matrix(depth, a.children[BOTTOM_RIGHT], r - mid, c - mid)
            end
        end
    end
end

function set_matrix(depth::Int, a::Union{Node,Nothing}, r::Int, c::Int, value::Float64)
    if depth == BLOCK_DEPTH
        if isnothing(a)
            a = LeafNode(Block())
        end
        a.block.data[r+1, c+1] = value
    else
        if isnothing(a)
            a = InternalNode()
        end
        depth -= 1
        mid = 1 << depth
        if r < mid
            if c < mid
                a.children[TOP_LEFT] = set_matrix(depth, a.children[TOP_LEFT], r, c, value)
            else
                a.children[TOP_RIGHT] = set_matrix(depth, a.children[TOP_RIGHT], r, c - mid, value)
            end
        else
            if c < mid
                a.children[BOTTOM_LEFT] = set_matrix(depth, a.children[BOTTOM_LEFT], r - mid, c, value)
            else
                a.children[BOTTOM_RIGHT] = set_matrix(depth, a.children[BOTTOM_RIGHT], r - mid, c - mid, value)
            end
        end
    end
    return a
end

function num_blocks(depth::Int, a::Union{Node,Nothing})
    if isnothing(a)
        return 0
    end
    if depth == BLOCK_DEPTH
        return 1
    end
    depth -= 1
    return sum(num_blocks(depth, child) for child in a.children)
end

function num_nonzeros(depth::Int, a::Union{Node,Nothing})
    if isnothing(a)
        return 0
    end
    if depth == BLOCK_DEPTH
        return count(!iszero, a.block.data)
    end
    depth -= 1
    return sum(num_nonzeros(depth, child) for child in a.children)
end


function mul_and_subT!(depth::Int, lower::Bool, a::Node, b::Node, r::Union{Node,Nothing})
    if depth == BLOCK_DEPTH
        r = isnothing(r) ? new_block_leaf() : r
        if lower
            block_schur_half!(r.block, a.block, b.block)
        else
            block_schur_full!(r.block, a.block, b.block)
        end
        return r
    end

    depth -= 1

    if (isnothing(r))
        r00, r01, r10, r11 = nothing, nothing, nothing, nothing
    else
        r00, r01, r10, r11 = r.children[TOP_LEFT], r.children[TOP_RIGHT], r.children[BOTTOM_LEFT], r.children[BOTTOM_RIGHT]
    end

    t1,t2,t3,t4 = nothing, nothing, nothing, nothing
    if !isnothing(a.children[TOP_LEFT]) && !isnothing(b.children[TRANS_TOP_LEFT])
        t1 = Threads.@spawn mul_and_subT!(depth, lower, a.children[TOP_LEFT], b.children[TRANS_TOP_LEFT], r00)
    end
    if !lower && !isnothing(a.children[TOP_LEFT]) && !isnothing(b.children[TRANS_TOP_RIGHT])
        t2 = Threads.@spawn mul_and_subT!(depth, false, a.children[TOP_LEFT], b.children[TRANS_TOP_LEFT], r01)
    end
    if !isnothing(a.children[BOTTOM_LEFT]) && !isnothing(b.children[TRANS_TOP_LEFT])
        t3 = Threads.@spawn mul_and_subT!(depth, false, a.children[BOTTOM_LEFT], b.children[TRANS_TOP_LEFT], r10)
    end
    if !isnothing(a.children[BOTTOM_LEFT]) && !isnothing(b.children[TRANS_TOP_RIGHT])
        t4 = Threads.@spawn mul_and_subT!(depth, lower, a.children[BOTTOM_LEFT], b.children[TRANS_TOP_RIGHT], r11)
    end
    if (t1 !== nothing)
        r00 = fetch(t1)
    end
    if (t2 !== nothing)
        r01 = fetch(t2)
    end
    if (t3 !== nothing)
        r10 = fetch(t3)
    end
    if (t4 !== nothing)
        r11 = fetch(t4)
    end

    t1,t2,t3,t4 = nothing, nothing, nothing, nothing
    if !isnothing(a.children[TOP_RIGHT]) && !isnothing(b.children[TRANS_BOTTOM_LEFT])
        t1 = Threads.@spawn mul_and_subT!(depth, lower, a.children[TOP_RIGHT], b.children[TRANS_BOTTOM_LEFT], r00)
    end
    if !lower && !isnothing(a.children[TOP_RIGHT]) && !isnothing(b.children[TRANS_BOTTOM_RIGHT])
        t2 = Threads.@spawn mul_and_subT!(depth, false, a.children[TOP_RIGHT], b.children[TRANS_BOTTOM_RIGHT], r01)
    end
    if !isnothing(a.children[BOTTOM_RIGHT]) && !isnothing(b.children[TRANS_BOTTOM_LEFT])
        t3 = Threads.@spawn mul_and_subT!(depth, false, a.children[BOTTOM_RIGHT], b.children[TRANS_BOTTOM_LEFT], r10)
    end
    if !isnothing(a.children[BOTTOM_RIGHT]) && !isnothing(b.children[TRANS_BOTTOM_RIGHT])
        t4 = Threads.@spawn mul_and_subT!(depth, lower, a.children[BOTTOM_RIGHT], b.children[TRANS_BOTTOM_RIGHT], r11)
    end

    if (t1 !== nothing)
        r00 = fetch(t1)
    end
    if (t2 !== nothing)
        r01 = fetch(t2)
    end
    if (t3 !== nothing)
        r10 = fetch(t3)
    end
    if (t4 !== nothing)
        r11 = fetch(t4)
    end

    if (isnothing(r))
        r = new_internal(r00, r01, r10, r11)
    else
        @assert(r.children[TOP_LEFT] === nothing || r.children[TOP_LEFT] == r00)
        @assert(r.children[TOP_RIGHT] === nothing || r.children[TOP_RIGHT] == r01)
        @assert(r.children[BOTTOM_LEFT] === nothing || r.children[BOTTOM_LEFT] == r10)
        @assert(r.children[BOTTOM_RIGHT] === nothing || r.children[BOTTOM_RIGHT] == r11)
        r.children[TOP_LEFT] = r00
        r.children[TOP_RIGHT] = r01
        r.children[BOTTOM_LEFT] = r10
        r.children[BOTTOM_RIGHT] = r11
    end
    return r
end

function backsub!(depth::Int, a::Node, l::Node)
    if depth == BLOCK_DEPTH
        block_backsub!(a.block, l.block)
        return a
    end

    depth -= 1

    a00, a01, a10, a11 = a.children[TOP_LEFT], a.children[TOP_RIGHT], a.children[BOTTOM_LEFT], a.children[BOTTOM_RIGHT]
    l00, l10, l11 = l.children[TOP_LEFT], l.children[BOTTOM_LEFT], l.children[BOTTOM_RIGHT]
    @assert(!isnothing(l00) && !isnothing(l11))
    t1,t2 = nothing, nothing
    if !isnothing(a00)
        t1 = Threads.@spawn backsub!(depth, a00, l00)
    end
    if !isnothing(a10)
        t2 = Threads.@spawn backsub!(depth, a10, l00)
    end
    if (t1 !== nothing)
        a00 = fetch(t1)
    end
    if (t2 !== nothing)
        a10 = fetch(t2)
    end


    t1,t2 = nothing, nothing
    if !isnothing(a00) && !isnothing(l10)
        t1 = Threads.@spawn mul_and_subT!(depth, false, a00, l10, a10)
    end
    if !isnothing(a10) && !isnothing(l10)
        t2 = Threads.@spawn mul_and_subT!(depth, false, a10, l10, a11)
    end
    if (t1 !== nothing)
        a01 = fetch(t1)
    end
    if (t2 !== nothing)
        a11 = fetch(t2)
    end


    t1,t2 = nothing, nothing
    if !isnothing(a01)
        t1 = Threads.@spawn backsub!(depth, a01, l11)
    end
    if !isnothing(a11)
        t2 = Threads.@spawn backsub!(depth, a11, l11)
    end
    if (t1 !== nothing)
        a01 = fetch(t1)
    end
    if (t2 !== nothing)
        a11 = fetch(t2)
    end

    a.children[TOP_LEFT] = a00
    a.children[TOP_RIGHT] = a01
    a.children[BOTTOM_LEFT] = a10
    a.children[BOTTOM_RIGHT] = a11
    return a
end

function cilk_cholesky!(depth::Int, a::Node)
    if depth == BLOCK_DEPTH
        block_cholesky!(a.block)
        return a
    end
    global last_node = (a, depth)
    depth -= 1

    a00 = a.children[TOP_LEFT]
    a10 = a.children[BOTTOM_LEFT]
    a11 = a.children[BOTTOM_RIGHT]
    @assert(!isnothing(a00))

    if (a10 === nothing)
        t = Threads.@spawn cilk_cholesky!(depth, a00)
        a11 = cilk_cholesky!(depth, a11)
        a00 = fetch(t)
    else
        a00 = cilk_cholesky!(depth, a00)
        a10 = backsub!(depth, a10, a00)
        a11 = mul_and_subT!(depth, true, a10, a10, a11)
        a11 = cilk_cholesky!(depth, a11)
    end
    a.children[TOP_LEFT] = a00
    a.children[BOTTOM_LEFT] = a10
    a.children[BOTTOM_RIGHT] = a11
    return a
end


function main()
    m_size = 500
    nonzeros = 1000
    depth = ceil(Int, log2(m_size))

    # Generate random matrix
    A = nothing
    for i in 1:m_size
        A = set_matrix(depth, A, i - 1, i - 1, 1.0)
    end

    for _ in 1:(nonzeros-m_size)
        r, c = 0, 0
        while true
            r = rand(0:(m_size-1))
            c = rand(0:(m_size-1))
            if r > c && get_matrix(depth, A, r, c) == 0.0
                break
            end
        end
        A = set_matrix(depth, A, r, c, 0.1)
    end

    # Extend to power of two m_size with identity matrix
    for i in m_size:(1<<depth)-1
        A = set_matrix(depth, A, i, i, 1.0)
    end

    R = copy_matrix(depth, A)

    input_blocks = num_blocks(depth, R)
    input_nonzeros = num_nonzeros(depth, R)

    start_time = time()
    R = cilk_cholesky!(depth, R)
    end_time = time()

    runtime_ms = (end_time - start_time) * 1000
    @printf("%.3f\n", runtime_ms / 1000)

    output_blocks = num_blocks(depth, R)
    output_nonzeros = num_nonzeros(depth, R)

    println("\nJulia Example: cholesky")
    println("Options: original m_size     = $m_size")
    println("         original nonzeros = $nonzeros")
    println("         input nonzeros    = $input_nonzeros")
    println("         input blocks      = $input_blocks")
    println("         output nonzeros   = $output_nonzeros")
    println("         output blocks     = $output_blocks")
end


function convert_to_matrix(a::AbstractMatrix, depth::Int)
    A = nothing
    for ind in eachindex(IndexCartesian(),a)
        A = set_matrix(depth, A, ind.I[1] - 1, ind.I[2] - 1, a[ind])
    end
    return A
end

function convert_to_julia(a::Node, depth::Int, m_size::Int)
    out = zeros(m_size, m_size)
    for i in 1:m_size, j in 1:m_size
        out[i,j] = get_matrix(depth, a, i-1, j-1)
    end
    return out
end
function make_spd(n)
    Q = rand(n,n)
    Q*Q'
end

function test(n)
    M = make_spd(n)
    cholesky(M)
    depth = ceil(Int, log2(n))
    m_size = n

    println("trying matrix with size $n and depth $depth")
    A = convert_to_matrix(M, depth)
    for i in (m_size + 1):(1<<depth)
        A = set_matrix(depth, A, i-1, i-1, 1.0)
    end
    R = copy_matrix(depth, A)
    global testing_m = A, M
    R = cilk_cholesky!(depth, R)
    L = convert_to_julia(R, depth, m_size)
    println("Error: ", norm(M - L*transpose(L)))
end
# main()
#This is a forward sub
function block_backsub!(B, L)
    n = size(B, 1)
    for j in 1:n
        for i in 1:n
            for k in 1:(i-1)
                B[i, j] -= L[i, k] * B[k, j]
            end
        B[i, j] /= L[i, i]
        end
    end
end

function forwardsubstitution!(L::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    n = size(L, 1)

    for j in 1:n  # Iterate over each column of B
        for i in 1:n
            sum = zero(T)
            for k in 1:i-1
                sum += L[i, k] * B[k, j]
            end
            B[i, j] = (B[i, j] - sum) / L[i, i]
        end
    end

    return B
end

function backsub2(B, U)
    @assert size(B) == size(U)
    n = size(B, 1)
    for i in 1:n
        for j in 1:n
            for k in 1:i
                B[j, i] -= U[i, k] * B[j, k]
            end
            B[j, i] /= U[i, i]
        end
    end
end