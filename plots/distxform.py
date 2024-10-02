import math

def gaussian_cdf(value, loc, scale):
    return 0.5 * (1 + torch.erf((value - loc) * (1 / scale) / math.sqrt(2)))

def laplace_icdf(value, loc, scale):
    term = value - 0.5
    return loc - scale / math.sqrt(2) * (term).sign() * torch.log1p(-2 * term.abs())

def laplace_cdf(value, loc, scale):
    return 0.5 - 0.5 * (value - loc).sign() * torch.expm1(-(value - loc).abs() / ( scale / math.sqrt(2)))

def gauss_laplace_compand(value, loc, scale):
    return laplace_cdf(laplace_icdf(gaussian_cdf(value, loc, scale), loc, scale), loc, 3 * scale)

def laplace_compand(value, loc, scale):
    return laplace_cdf(value, loc, 3 * scale)

def gauss_compand(value, loc, scale):
    return gaussian_cdf(value, loc, math.sqrt(3) * scale)
