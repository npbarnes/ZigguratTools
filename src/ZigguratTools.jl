module ZigguratTools

using Roots
using Plots

export buildziggurat, buildziggurat!
export searchziggurat
export sampleziggurat, sampleziggurat!, symmetricsampleziggurat, symmetricsampleziggurat!
export plotziggurat
export ZigguratSampler, xvalues, yvalues, xyvalues, layerarea

# TODO: Ziggurat should be a sampler similar to samplers from Random.jl and Distributions.jl
struct ZigguratSampler{X,Y,XY,T,F}
    x::Vector{X}
    y::Vector{Y}
    A::XY
    tailsample::T
    pdf::F
end

function ZigguratSampler(f, finv, F, N, tailsample, method=Bisection(); xtype=Float64)
    x,y = searchziggurat(f, finv, F, N, method; xtype)
    A = x[1] * (y[2] - y[1])
    ZigguratSampler(x, y, A, tailsample, f)
end

xvalues(zs::ZigguratSampler) = zs.x
yvalues(zs::ZigguratSampler) = zs.y
xyvalues(zs::ZigguratSampler) = (zs.x, zs.y)
layerarea(zs::ZigguratSampler) = zs.A
pdf(zs::ZigguratSampler, x) = zs.f(x)
tailsample(zs::ZigguratSampler) = zs.tailsample(xvalues(zs)[1])

# TODO: Implement plotting as a Plots.jl recipe
# TODO: Put the plotting stuff in an extension
function plotziggurat(zs::ZigguratSampler)
    x,y = xyvalues(zs)

    p = plot([0, x[1]], [y[1], y[1]], color=:black, legend=false)
    scatter!(p, [x[1]], [y[1]], color=:black)
    for i in eachindex(x)[2:end]
        plot!(p, [0, x[i-1], x[i-1]], [y[i], y[i], y[i-1]], color=:black)
        plot!(p, [x[i], x[i]], [y[i], y[i-1]], color=:black, ls=:dash)
        scatter!(p, [x[i]], [y[i]], color=:black)
    end
    xl = xlims(p)
    yl = ylims(p)
    xlims!(0,1.5xl[2])
    ylims!(0,yl[2])

    plot!(p, zs.f, color=:blue, lw=2)

    p
end

"""
    buildziggurat(f, finv, F, N, x1; f0=f(zero(x1)))

Build a ziggurat for f with N levels given a known x1. Returns a tuple of vectors for
the x values and the y values respectively. An error is thrown if the ziggurat is
invalid. For example, when the N'th level fails to cover f, or if the N'th level is not
close to f(0).
"""
function buildziggurat(f, finv, F, N, x1; f0=f(zero(x1)))
    x = Vector{typeof(x1)}(undef, N)
    y = Vector{typeof(f0)}(undef, N)
    buildziggurat!(x, y, f, finv, F, x1; f0)
end

"""
    buildziggurat!(x, y, f, finv, F, x1; f0=f(zero(eltype(x))))

Build a ziggurat for f and put the x and y values into the arguments x and y. The arguments
x and y should be Vectors with the same length. An error is thrown if the ziggurat is
invalid. For example, when the N'th level fails to cover f, or if the N'th level is not 
close to f(0).
"""
function buildziggurat!(x, y, f, finv, F, x1; f0=f(zero(x1)))
    # Check arguments
    if x1 <= zero(x1)
        throw(ArgumentError("x1 must be a real positive number."))
    end
    if length(x) != length(y)
        throw(ArgumentError("the lengths of x and y must be the same"))
    end

    # Do the actual work
    zig = _buildziggurat!(x, y, f, finv, F, x1; f0)
    
    # Sanity checks
    if zig === nothing
        error("failed to build a covering ziggurat. The starting value, x1=$x1, is too small for N=$N.")
    end

    x,y = zig
    if y[end] < f0
        error("failed to buld a covering ziggurat. The starting value, x1=$x1, is too large for N=$N.")
    elseif y[end] ≉ f0
        error("failed to build an accurate ziggurat. y[N] = $(y[N]) is not close to f(0) = $f0.\
        The starting value, x1=$x1, is too small for N=$N.")
    end

    zig
end

# TODO: add fallback for F(x) = quadgk(f, -Inf, x)
# TODO: add interface for supplying F or 1-F
# TODO: add fallback for 1-F(x) = quadgk(f, x, Inf)
function _buildziggurat!(x, y, f, finv, F, x1; f0)
    N = length(x) # length(x) == length(y)
    x[1] = x1
    y[1] = f(x1)
    A = x[1] * y[1] + (1 - F(x[1])) # TODO: support unnormalized functions

    y[2] = y[1] + A/x[1]
    for i in 2:N-1
        if y[i] >= f0
            # Building the ziggurat failed
            return nothing 
        end
        x[i] = finv(y[i])
        y[i+1] = y[i] + A/x[i]
    end

    # The sampling algorithm requires x[N] = 0.
    x[N] = zero(eltype(x))
    
    x,y
end


"""
    searchziggurat(f, finv, F, N, method=Bisection(); xytpe=Float64)

Build a ziggurat for f when x1 is unknown by searching for the correct value.
Returns a tuple of Vectors for the x and y values of the ziggurat respectively.
The method argument may be any of the derivative free methods provided by
Roots.jl; Bisection() is the default because it is guarenteed to converge.
"""
function searchziggurat(f, finv, F, N, method=Bisection(); xtype=Float64)
    f0 = f(zero(xtype))
    x = Vector{xtype}(undef, N)
    y = Vector{typeof(f0)}(undef, N)

    "Try to build a ziggurat and return a value that represents error"
    function attemptziggurat!(x1)
        zig = _buildziggurat!(x, y, f, finv, F, x1; f0)
        if zig === nothing
            return Inf
        end

        x,y = zig
        y[N] - f0
    end

    tracker = Roots.Tracks()
    xstar = find_zero(attemptziggurat!, (0, Inf), method; tracks=tracker)

    # y[N] needs to be either exact or a slight over estimate
    if tracker.convergence_flag !== :exact_zero
        xstar = tracker.abₛ[end][1]
    end
    #TODO handle non-convergence.

    buildziggurat!(x, y, f, finv, F, xstar; f0)
end

function sampleziggurat!(out, zs::ZigguratSampler)
    for i in eachindex(out)
        out[i] = sampleziggurat(zs)
    end
    out
end

function sampleziggurat(zs::ZigguratSampler, N)
    out = Vector{Float64}(undef, N)
    sampleziggurat!(out, zs)
end

"""
    sampleziggurat(zs::ZigguratSampler)

Sample a point from the distribution using the ziggurat method.
"""
function sampleziggurat(zs::ZigguratSampler)
    N = length(zs.x)

    while true
        l = rand(1:N)
        if l == 1 # Baselayer
            x0 = zs.A/zs.y[1]
            x = x0*rand()
            if x < zs.x[1]
                return x
            else
                return tailsample(zs)
            end
        else # all other layers
            x = zs.x[l-1]*rand()
            if x < zs.x[l]
                return x
            else
                y = (zs.y[l] - zs.y[l-1])*rand() + zs.y[l-1]
                if y < zs.pdf(x)
                    return x
                end
            end
        end
    end
end

function symmetricsampleziggurat(zs::ZigguratSampler)
    x = sampleziggurat(zs)
    ifelse(rand(Bool), x, -x)
end

function symmetricsampleziggurat(zs::ZigguratSampler, N)
    out = Vector{Float64}(undef, N)
    symmetricsampleziggurat!(out, zs)
end

function symmetricsampleziggurat!(out, zs::ZigguratSampler)
    for i in eachindex(out)
        out[i] = symmetricsampleziggurat(zs)
    end
    out
end

end # module ZigguratTools
