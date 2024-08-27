using LinearAlgebra
using Random
using Printf

const BLOCK_DEPTH = 2
const BLOCK_SIZE = 1 << BLOCK_DEPTH

struct Block
    data::Matrix{Float64}
end

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
    B.data .-= A.data * C.data'
end

function block_schur_half!(B::Block, A::Block, C::Block)
    for i in 1:BLOCK_SIZE, j in 1:i
        for k in 1:BLOCK_SIZE
            B.data[i, j] -= A.data[i, k] * C.data[j, k]
        end
    end
end

function block_backsub!(B::Block, U::Block)
    for i in 1:BLOCK_SIZE, j in 1:BLOCK_SIZE
        for k in 1:i-1
            B.data[j, i] -= U.data[i, k] * B.data[j, k]
        end
        B.data[j, i] /= U.data[i, i]
    end
end

function block_cholesky!(B::Block)
    for k in 1:BLOCK_SIZE
        if B.data[k, k] < 0
            error("Matrix is not numerically stable")
        end
        B.data[k:end, k] ./= sqrt(B.data[k, k])
        for j in k+1:BLOCK_SIZE, i in j:BLOCK_SIZE
            B.data[i, j] -= B.data[i, k] * B.data[j, k]
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
    r00 = Threads.@spawn copy_matrix(depth, a.children[1])
    r01 = Threads.@spawn copy_matrix(depth, a.children[2])
    r10 = Threads.@spawn copy_matrix(depth, a.children[3])
    r11 = copy_matrix(depth, a.children[4])

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
                return get_matrix(depth, a.children[1], r, c)
            else
                return get_matrix(depth, a.children[2], r, c - mid)
            end
        else
            if c < mid
                return get_matrix(depth, a.children[3], r - mid, c)
            else
                return get_matrix(depth, a.children[4], r - mid, c - mid)
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
                a.children[1] = set_matrix(depth, a.children[1], r, c, value)
            else
                a.children[2] = set_matrix(depth, a.children[2], r, c - mid, value)
            end
        else
            if c < mid
                a.children[3] = set_matrix(depth, a.children[3], r - mid, c, value)
            else
                a.children[4] = set_matrix(depth, a.children[4], r - mid, c - mid, value)
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
    r = isnothing(r) ? InternalNode() : r

    Threads.@sync begin
        if !isnothing(a.children[1]) && !isnothing(b.children[1])
            Threads.@spawn mul_and_subT!(depth, lower, a.children[1], b.children[1], r.children[1])
        end
        if !lower && !isnothing(a.children[1]) && !isnothing(b.children[2])
            Threads.@spawn mul_and_subT!(depth, false, a.children[1], b.children[2], r.children[2])
        end
        if !isnothing(a.children[3]) && !isnothing(b.children[1])
            Threads.@spawn mul_and_subT!(depth, false, a.children[3], b.children[1], r.children[3])
        end
        if !isnothing(a.children[3]) && !isnothing(b.children[2])
            Threads.@spawn mul_and_subT!(depth, lower, a.children[3], b.children[2], r.children[4])
        end
    end

    Threads.@sync begin
        if !isnothing(a.children[2]) && !isnothing(b.children[3])
            Threads.@spawn mul_and_subT!(depth, lower, a.children[2], b.children[3], r.children[1])
        end
        if !lower && !isnothing(a.children[2]) && !isnothing(b.children[4])
            Threads.@spawn mul_and_subT!(depth, false, a.children[2], b.children[4], r.children[2])
        end
        if !isnothing(a.children[4]) && !isnothing(b.children[3])
            Threads.@spawn mul_and_subT!(depth, false, a.children[4], b.children[3], r.children[3])
        end
        if !isnothing(a.children[4]) && !isnothing(b.children[4])
            Threads.@spawn mul_and_subT!(depth, lower, a.children[4], b.children[4], r.children[4])
        end
    end

    return r
end

function backsub!(depth::Int, a::Node, l::Node)
    if depth == BLOCK_DEPTH
        block_backsub!(a.block, l.block)
        return a
    end

    depth -= 1

    Threads.@sync begin
        if !isnothing(a.children[1])
            Threads.@spawn backsub!(depth, a.children[1], l.children[1])
        end
        if !isnothing(a.children[3])
            Threads.@spawn backsub!(depth, a.children[3], l.children[1])
        end
    end

    Threads.@sync begin
        if !isnothing(a.children[1]) && !isnothing(l.children[3])
            Threads.@spawn mul_and_subT!(depth, false, a.children[1], l.children[3], a.children[2])
        end
        if !isnothing(a.children[3]) && !isnothing(l.children[3])
            Threads.@spawn mul_and_subT!(depth, false, a.children[3], l.children[3], a.children[4])
        end
    end
    Threads.@sync begin
        if !isnothing(a.children[2])
            Threads.@spawn backsub!(depth, a.children[2], l.children[4])
        end
        if !isnothing(a.children[4])
            Threads.@spawn backsub!(depth, a.children[4], l.children[4])
        end
    end
    return a
end

function cilk_cholesky!(depth::Int, a::Node)
    if depth == BLOCK_DEPTH
        block_cholesky!(a.block)
        return a
    end

    depth -= 1

    if isnothing(a.children[3])
        t = Threads.@spawn cilk_cholesky!(depth, a.children[1])
        a.children[4] = cilk_cholesky!(depth, a.children[4])
        a.children[1] = fetch(a.children[1])
    else
        a.children[1] = cilk_cholesky!(depth, a.children[1])
        a.children[3] = backsub!(depth, a.children[3], a.children[1])
        a.children[4] = mul_and_subT!(depth, true, a.children[3], a.children[3], a.children[4])
        a.children[4] = cilk_cholesky!(depth, a.children[4])
    end

    return a
end

function convert_to_julia(a::Node, m_size::Int)
    if isnothing(a)
        return nothing
    end

    if isa(a, LeafNode)
        return a.block.data
    end

    return hcat(
        vcat(convert_to_julia(a.children[1]), convert_to_julia(a.children[2])),
        vcat(convert_to_julia(a.children[3]), convert_to_julia(a.children[4]))
    )
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

main()