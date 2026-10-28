"""Unit conversions. Internals are SI; configs keep the units ND200 uses."""

MPH_TO_MS = 0.44704
LBF_TO_N = 4.4482216152605
IN_TO_M = 0.0254
G_TO_KG = 1e-3


def mph_to_ms(v):
    return v * MPH_TO_MS


def ms_to_mph(v):
    return v / MPH_TO_MS


def lbf_to_n(f):
    return f * LBF_TO_N


def n_to_lbf(f):
    return f / LBF_TO_N


def in_to_m(x):
    return x * IN_TO_M
