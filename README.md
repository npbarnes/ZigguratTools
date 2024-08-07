# ZigguratTools
[![CI](https://github.com/npbarnes/ZigguratTools/actions/workflows/CI.yml/badge.svg)](https://github.com/npbarnes/ZigguratTools/actions/workflows/CI.yml)

This package will be a collection of tools for generating fast ziggurat-type random samplers for a large class of probability distributions. **Currently in development.**

## Goals

This package aims to make using the Ziggurat Method[^1] for random variate generation as simple as possible. Julia provides implementations of the ziggurat algorithm for normal and exponential distributions, but the algorithm could be adapted to a large class of distributions. The annoying part is generating the tables of values used by the algorithm. 

The table generation algorithm requires several inputs: pdf, inverse pdf, cdf, and mode. In addition, a fallback algorithm for the tail is needed (most likely using an inverse cdf). Having the user figure all that out is too much to ask, in my opinion. I want to automate it as much as possible. I plan to use root finding, autodifferentiation, and numerical integration to compute the inverse pdf, cdf, mode, and inverse cdf. Ideally, I'd like to be able to provide a pdf and get a sampler back that implements a ziggurat algorithm with performance similar to Julia's `randn` and `randexp` functions.  I expect to be able to achieve that high level of performance for sampling, but the ziggurat generation will be a potentially slow operation. Therefore, this algorithm will only make sense in contexts where many samples are needed from a fixed distribution.

At first, I am focusing on monotonic distributions with finite density. That includes functions that are not strictly monotonic. The ziggurat algorithm is usually applied to unimodal distributions by randomly selecting a sign, but I believe I can extend that to piecewise monotonic distributions using an alias table. In the future, I may also implement the Generalized Ziggurat Method from Jalavand and Charsooghi[^2] to support distributions with unbounded densities.

## Status

I'm currently working on monotonic distributions. They will be the foundation of more complicated distributions, so it is important to get them right. Right now I can often make simple ziggurats and sample from them correctly, but there remain a lot of dangerous corner cases. For example, I started with a few manually written inverse pdfs, but I've found that floating point rounding can cause problems in specific circumstances. Counterintuitively, I think it's better to compute the generalized inverse using a bisection method since it can make certain guarantees that I don't get from floating point algebra. I would have liked to use Roots.jl or NonlinearSolve.jl for this problem, but they're inappropriate for this job because they return if they find an exact zero. However, I'm not satisfied with any x that satisfies the equation. I need the largest x that satisfies the equation. That's a slightly different problem. Fortunately, a bisection search is not a complicated algorithm to implement.

[^1]: Marsaglia, G., & Tsang, W. W. (2000). The Ziggurat Method for Generating Random Variables. Journal of Statistical Software, 5(8), 1–7. https://doi.org/10.18637/jss.v005.i08
[^2]: Jalalvand, M., & Charsooghi, M. A. (2018). Generalized ziggurat algorithm for unimodal and unbounded probability density functions with Zest. arXiv preprint arXiv:1810.04744.
