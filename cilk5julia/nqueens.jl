using Printf
using Base.Threads

function ok(n::Int, a::Vector{Int})
    for i in 1:n
        p = a[i]
        for j in (i+1):n
            q = a[j]
            if q == p || q == p - (j - i) || q == p + (j - i)
                return false
            end
        end
    end
    return true
end


function nqueens(n::Int, j::Int, a::Vector{Int})
    if n == j
        return 1
    end

    tasks = Memory{Task}(undef, n)

    for i in 1:n
        b = copy(a)
        b[j+1] = i
        if ok(j + 1, b)
            tasks[i] = Threads.@spawn nqueens(n, j + 1, b)
        end
    end
    tot = 0
    for i in eachindex(tasks)
        if isassigned(tasks, i)
            tot += fetch(tasks[i])::Int
        end
    end
    return tot
end

function main()
    n = 13
    if length(ARGS) > 0
        n = parse(Int, ARGS[1])
        @printf("Running with n = %d.\n", n)
    else
        println("Usage: julia nqueens.jl <n>")
        println("Using default board size, n = 13.")
    end
    println("warmup !Running nqueens...")
    nqueens(n, 0, zeros(Int, n)) #warmup
    println("warmup !Finished running nqueens.")
    a = zeros(Int, n)

    # start_time = time_ns()
    println("Running nqueens...")
    @time begin
        t = Threads.@spawn nqueens(n, 0, a)
        res = fetch(t)
    end
    println("Finished running nqueens.")
    # end_time = time_ns()

    # runtime_s = end_time - start_time
    # runtime_s /= 1e9
    # @printf("%.3f\n", runtime_s)

    if res == 0
        println("No solution found.")
    else
        @printf("Total number of solutions: %d\n", res)
    end
end

function test(n, verbose=true)
    a = zeros(Int, n)

    # start_time = time_ns()
    @time begin
        t = Threads.@spawn nqueens(n, 0, a)
        res = fetch(t)
    end
    # end_time = time_ns()

    runtime_s = end_time - start_time
    runtime_s /= 1e9
    if verbose
        @printf("%.3f\n", runtime_s)
        if res == 0
            println("No solution found.")
        else
            @printf("Total number of solutions: %d\n", res)
        end
    end
end

if !isinteractive()
    main()
end
