abstract type MonotonicZiggurat{X} <: Ziggurat{X} end

struct BoundedMonotonicZiggurat{X,Y,F<:Function} <: MonotonicZiggurat{X}
    x::Vector{X}
    y::Vector{Y}
    pdf::F
    modalboundary::X
end

struct UnboundedMonotonicZiggurat{X,Y,F<:Function,TS} <: MonotonicZiggurat{X}
    x::Vector{X}
    y::Vector{Y}
    pdf::F
    modalboundary::X
    tailsampler::TS
end

"""
    monotonic_ziggurat(N, L, R, pdf, ipdf)

Returns a BoundedMonotonicZiggurat that implements the sampler interface to rand().

# Example
```julia-repl
julia> z = monotonic_ziggurat(8, 0.0, 5.0, x -> exp(-x^2), y -> √(-log(y)))
BoundedMonotonicZiggurat{...}(...)

julia> import Random; Random.seed!(1234);

julia> rand(z)
0.6851716653106885
```
"""
function monotonic_ziggurat(N, L, R::Number, pdf, ipdf)
    if N < 1
        throw(DomainError(N, "N must be a positive integer."))
    end
    boundaries = (L, R)
    boundarypdf = pdf.(boundaries)
    argminboundary = boundaries[argmin(boundarypdf)]
    modalboundary = boundaries[argmax(boundarypdf)]

    x, y = search(N, modalboundary, argminboundary, pdf, ipdf)
    BoundedMonotonicZiggurat(x, y, pdf, modalboundary)
end

"""
    monotonic_ziggurat(N, modalboundary, tailarea, pdf, ipdf, tailsampler)

Returns an UnboundedMonotonicZiggurat that implements the sampler interface to
rand(). The arguments `tailarea` and `tailsampler` are functions. `tailarea(x)`
should return the area of the tail starting at `x`, and `tailsampler(x)` should
return a sampler that samples from the tail (to be used as the fallback
algorithm).

# Example
```julia-repl
julia> using ZigguratTools, Distributions

julia> dist = truncated(Normal(), lower=0.0)
Truncated...

julia> z = monotonic_ziggurat(
    8,
    0.0,
    x -> ccdf(dist, x),
    x -> pdf(dist, x),
    y -> ipdf_right(dist, y), # ipdf_right is provided by ZigguratTools.jl
    x -> sampler(truncated(dist, lower=x))
)
UnboundedMonotonicZiggurat...

julia> import Random; Random.seed!(1234);

julia> rand(z)
0.8937377790003791
```
"""
function monotonic_ziggurat(N, modalboundary, tailarea::Function, pdf, ipdf, tailsampler)
    if N < 1
        throw(DomainError(N, "N must be a positive integer."))
    end
    x, y = search(N, modalboundary, tailarea, pdf, ipdf)
    UnboundedMonotonicZiggurat(x, y, pdf, modalboundary, tailsampler(x[2]))
end

"""
    search(N, modalboundary, argminboundary::Number, pdf, ipdf)
    search(N, modalboundary, tailarea::Function, pdf, ipdf)

Returns x and y arrays for a correct and optimal ziggurat with N layers or
nothing if the search fails. Note that a ziggurat with N layers satisfies
length(x) == length(y) == N+1.
"""
function search(N, modalboundary, tail, pdf, ipdf)
    modalpdf = pdf(modalboundary)
    x = Vector{typeof(float(modalboundary))}(undef, N + 1)
    y = Vector{typeof(modalpdf)}(undef, N + 1)

    function attemptziggurat!(y2)
        zig = build!(x, y, y2, modalboundary, tail, ipdf, modalpdf)
        if zig === nothing
            return Inf
        end

        x, y = zig
        y[end] - modalpdf
    end

    # TODO: Roots.Tracks may change between versions, so we should use an alternative (SciML?)
    tracker = Roots.Tracks()
    ystar = find_zero(
        attemptziggurat!,
        (nextfloat(zero(modalpdf)), modalpdf),
        Bisection();
        tracks = tracker
    )

    # y[N] needs to be either exact or a slightly over
    if tracker.convergence_flag !== :exact_zero
        ystar = tracker.abₛ[end][2]
    end
    #TODO handle non-convergence. Bisection is guarenteed to converge, but not all
    #algorithms are.

    build!(x, y, ystar, modalboundary, tail, ipdf, modalpdf)
end

# The caller of build!() is responsible for ensuring consistancy of the build
# arguments. I.e.,
# 1) length(x) == length(y) >= 1
# 2) zero(y2) < y2 <= pdf(boundary)
# 3) boundarypdf = pdf(boundary)
# 4) the pdf is monotonic
# 5) the ipdf is the gerneralized inverse of pdf (i.e. ipdf(y) is the largest x
#       such that pdf(x) >= y for decreasing pdfs, and ipdf(y) is the smallest x
#       such that pdf(x) >= y for increasing pdfs)

# Bounded support
function build!(x, y, y2, modalboundary, argminboundary::Number, ipdf, modalpdf)
    initialize!(y, y2)

    A = layerarea(y[2], modalboundary, argminboundary)
    x[1] = argminboundary

    finalize!(x, y, modalboundary, A, ipdf, modalpdf)
end

# Unbounded support
function build!(x, y, y2, modalboundary, tailarea::Function, ipdf, modalpdf)
    initialize!(y, y2)

    x2 = ipdf(y2)
    if isinf(x2)
        # TODO: find a more consistant way for build!() to indicate when y2 is
        # too small or too large.

        # y2 is too small
        y[end] = zero(eltype(y))
        return x, y
    end
    A = layerarea(y[2], x2, modalboundary, tailarea)

    if x2 == modalboundary
        s = sign(ipdf(y2 / 2) - modalboundary)
    else
        s = sign(x2 - modalboundary)
    end

    x[1] = modalboundary + s * A / y[2]

    finalize!(x, y, modalboundary, A, ipdf, modalpdf)
end

function initialize!(y, y2)
    y[1] = zero(eltype(y))
    y[2] = y2
end

# Bounded support
layerarea(y2, modalboundary, argminboundary) = abs(argminboundary - modalboundary) * y2

# Unbounded support
function layerarea(y2, x2, modalboundary, tailarea::Function)
    if y2 == zero(y2)
        return zero(x2 * y2)
    end
    abs(x2 - modalboundary) * y2 + tailarea(x2)
end

function finalize!(x, y, modalboundary, A, ipdf, modalpdf)
    for i in eachindex(x)[(begin + 1):(end - 1)]
        if y[i] >= modalpdf
            # failed to build ziggurat, y2 is too large.
            return nothing
        end
        x[i] = ipdf(y[i])
        y[i + 1] = A / abs(x[i] - modalboundary) + y[i]
    end

    if y[end] <= modalpdf
        x[end] = ipdf(y[end])
    else
        x[end] = modalboundary
    end

    x, y
end

## Sampling
function Base.rand(rng::AbstractRNG, z::MonotonicZiggurat)
    N = length(z.x) - 1 # number of layers

    while true
        l = rand(rng, 1:N)
        x = (z.modalboundary - z.x[l]) * rand(rng) + z.x[l]

        if between(z.x[l + 1], z.modalboundary, x)
            return x
        else
            sp = slowpath(rng, z, l, x)
            if sp !== nothing
                return sp
            end
        end
    end
end

function simple_rejection(rng, z, l, x)
    y = (z.y[l + 1] - z.y[l]) * rand(rng) + z.y[l]
    if y < z.pdf(x)
        return x
    end

    nothing
end

function slowpath(rng, z::UnboundedMonotonicZiggurat, l, x)
    if l == 1
        return rand(rng, z.tailsampler)
    end

    simple_rejection(rng, z, l, x)
end

slowpath(rng, z::BoundedMonotonicZiggurat, l, x) = simple_rejection(rng, z, l, x)
