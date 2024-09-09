
const KILO = 1024
const MERGESIZE = 2 * KILO
const QUICKSIZE = 2 * KILO
const INSERTIONSIZE = 20

function med3(a::T, b::T, c::T) where {T}
    if a < b
        return b < c ? b : a < c ? c : a
    else
        return b > c ? b : a > c ? c : a
    end
end
function choose_pivot(a::AbstractVector{T}) where {T}
    return med3(a[1], a[length(a) ÷ 2], a[length(a)])
end

function seqpart(a::AbstractVector{T}, start::Int, final::Int) where {T}
    curr_start = start
    curr_final = final
    p = choose_pivot(@view(a[start:final]))
    while true
        while a[curr_final] > p
            curr_final -= 1
        end
        while a[curr_start] < p
            curr_start += 1
        end
        if curr_start >= curr_final
            if (curr_final < final)
                return curr_final
            else
                return curr_final - 1
            end
        end
        a[curr_start], a[curr_final] = a[curr_final], a[curr_start]
        curr_start += 1
        curr_final -= 1
    end
end

function seqinsertion(a::AbstractVector{T}) where {T}
    for i in 2:length(a)
        j = i
        while j > 1 && a[j] < a[j-1]
            a[j], a[j-1] = a[j-1], a[j]
            j -= 1
        end
    end
end

function seqquick(a::AbstractVector{T}) where {T}
    seqquick(a, 1, length(a))
end

function seqquick(a::AbstractVector{T}, start::Int, final::Int) where {T}
    init_start = start
    # @show "before" a[init_start:final]
    while final - start > INSERTIONSIZE
        # @show start, final
        p = seqpart(a, start, final)
        # @show p
        seqquick(a, start, p)
        start = p + 1
    end
    seqinsertion(@view(a[start:final]))
    # @show "after" a[init_start:final]
    @assert(issorted(@view(a[init_start:final])))
end

# return index of first element smaller or equal to val
function binsearch(a::AbstractVector{T}, val::T) where {T}
    lo = 1
    hi = length(a)
    while lo < hi
        mid = (lo + hi) ÷ 2
        if a[mid] <= val
            lo = mid + 1
        else
            hi = mid
        end
    end
    if a[lo] > val
        return lo - 1
    end
    return lo
end


function seqmerge(a::AbstractVector{T}, b::AbstractVector{T}, dest::AbstractVector{T}) where {T}
    @assert length(a) + length(b) == length(dest)
    i = 1
    j = 1
    k = 1
    while i <= length(a) && j <= length(b)
        if a[i] <= b[j]
            dest[k] = a[i]
            i += 1
        else
            dest[k] = b[j]
            j += 1
        end
        k += 1
    end
    while i <= length(a)
        dest[k] = a[i]
        i += 1
        k += 1
    end
    while j <= length(b)
        dest[k] = b[j]
        j += 1
        k += 1
    end
end

function cilkmerge(A::AbstractVector{T}, B::AbstractVector{T}, dest::AbstractVector{T}) where {T}
    #    /*
#     /*
#     * Cilkmerge: Merges range [low1, high1] with range [low2, high2]
#     * into the range [lowdest, ...]
#     */

#    /*
#     * We want to take the middle element (indexed by split1) from the
#     * larger of the two arrays.  The following code assumes that split1
#     * is taken from range [low1, high1].  So if [low1, high1] is
#     * actually the smaller range, we should swap it with [low2, high2]
#     */
    @assert length(A) + length(B) == length(dest)
    if (length(A) > length(B))
        large = A
        small = B
    else
        large = B
        small = A
    end

    if length(small) == 0
        copyto!(dest, large)
        return
    end

    if length(large) < MERGESIZE
        seqmerge(A, B, dest)
        return
    end

    split1 = length(large) ÷ 2

    split2 = binsearch(small,large[split1])
    dest[split1+split2] = large[split1]


    @sync begin
        Threads.@spawn cilkmerge(@view(large[1:(split1-1)]), @view(small[1:split2]), @view(dest[1:(split1+split2-1)]))
        Threads.@spawn cilkmerge(@view(large[(split1+1):end]), @view(small[(split2+1):end]), @view(dest[(split1+split2+1):end]))
    end

end

function cilksort(low::AbstractVector{T}, buff::AbstractVector{T}) where T
    # * divide the input in four parts of the same size (A, B, C, D)
    # * Then:
    # *   1) recursively sort A, B, C, and D (in parallel)
    # *   2) merge A and B into tmp1, and C and D into tmp2 (in parallel)
    # *   3) merbe tmp1 and tmp2 into the original array
    # */
    @assert length(low) == length(buff)
    if length(low) <= QUICKSIZE
        return seqquick(low)
    end
    size = length(low)
    A = @view(low[1:(size÷4)])
    B = @view(low[(size÷4+1):(size÷2)])
    C = @view(low[(size÷2+1):(3*size÷4)])
    D = @view(low[(3*size÷4+1):size])
    tmpA = @view(buff[1:(size÷4)])
    tmpB = @view(buff[(size÷4+1):(size÷2)])
    tmpC = @view(buff[(size÷2+1):(3*size÷4)])
    tmpD = @view(buff[(3*size÷4+1):size])
    tmp1 = @view(buff[1:(size÷2)])
    tmp2 = @view(buff[(size÷2+1):size])
    @sync begin
        Threads.@spawn cilksort(A, tmpA)
        Threads.@spawn cilksort(B, tmpB)
        Threads.@spawn cilksort(C, tmpC)
        Threads.@spawn cilksort(D, tmpD)
    end
    @sync begin
        Threads.@spawn cilkmerge(A, B, tmp1)
        Threads.@spawn cilkmerge(C, D, tmp2)
    end
    cilkmerge(tmp1, tmp2, low)

    return low
end

