"""Impact simulation: build the node/element chain for a layer stack and run the compiled RK4 kernel.

See kernels.py for the physics. Everything here is bookkeeping: stack -> arrays, per-sample
ball/velocity/temperature arrays in, CFC120-filtered load-cell traces out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from chestforge import SIM_LABEL
from chestforge.config import load_nd200, load_nts
from chestforge.core import kernels
from chestforge.core.filters import cfc_filter
from chestforge.core.padding import FoamLayer, PlateLayer, Stack
from chestforge.core.projectile import Projectile
from chestforge.core.surrogate import NTS, silhouette_radius_m
from chestforge.units import n_to_lbf

FAR = 1.0e3  # m, "plate edge is nowhere near"
PRE_TRIGGER_S = 0.005  # zero lead-in before contact when filtering


@dataclass(frozen=True)
class ChainArrays:
    node_mass: np.ndarray
    contact_node: int
    e_top: np.ndarray
    e_bot: np.ndarray
    e_par: np.ndarray
    e_h: np.ndarray
    l_kind: np.ndarray
    l_ref: np.ndarray
    pl_D: np.ndarray
    pl_reach: np.ndarray
    backing_mass: float
    foams: tuple            # FoamLayer per element index


@dataclass(frozen=True)
class Rig:
    """What the stack sits on and what the load cell measures.

    m2/k2/c2: load-cell side (loading surface on its spring), m3: body behind it (free),
    k_found: elastic foundation under plates (N/m^3), r_share: radius of the instrumented area
    (force outside it bypasses the load cell), gain: data-acquisition gain."""
    name: str
    m2: float
    m3: float
    k2: float
    c2: float
    gain: float
    k_found: float
    r_share: float

    @classmethod
    def nts(cls, nts: NTS) -> "Rig":
        return cls("nts", nts.loading_surface_mass_kg, nts.back_mass_kg, nts.skin_stiffness_N_per_m,
                   nts.skin_damping_Ns_per_m, nts.gain, nts_foundation_modulus(nts), silhouette_radius_m())

    @classmethod
    def drop_tower(cls, load_cell_cap_kg: float = 0.1, stiffness_N_per_m: float = 5.0e7,
                   damping_ratio: float = 0.05) -> "Rig":
        """Pads on a rigid, fully instrumented anvil (load cell under the whole sample)."""
        c = 2 * damping_ratio * math.sqrt(stiffness_N_per_m * load_cell_cap_kg)
        return cls("drop_tower", load_cell_cap_kg, 1.0e4, stiffness_N_per_m, c, 1.0, 1.0e12, 1.0e3)


def nts_foundation_modulus(nts: NTS) -> float:
    """NTS skin as an elastic foundation, N/m^3: k2 spread over the silhouette (config/nts.yaml)."""
    return nts.skin_stiffness_N_per_m / (math.pi * silhouette_radius_m() ** 2)


def build_chain(stack: Stack, ball_radius_m: float, k_nts: float,
                plate_reach_m: Sequence[float] | None = None) -> ChainArrays:
    """Turn a stack (ball side first) into kernel arrays. Plates are rigid nodes; foams are elements."""
    node_count = 0
    current = None
    e_top, e_bot, foams = [], [], []
    l_kind, l_ref = [], []
    plates, plate_node = [], []
    for lay in stack:
        if current is None:
            current, node_count = node_count, node_count + 1
        if isinstance(lay, FoamLayer):
            l_kind.append(kernels.KIND_FOAM)
            l_ref.append(len(foams))
            e_top.append(current)
            current, node_count = node_count, node_count + 1
            e_bot.append(current)
            foams.append(lay)
        elif isinstance(lay, PlateLayer):
            l_kind.append(kernels.KIND_PLATE)
            l_ref.append(len(plates))
            plates.append(lay)
            plate_node.append(current)
        else:
            raise TypeError(f"unknown layer {lay!r}")

    # the deepest node is the backing (index -1)
    backing = current
    remap = lambda k: -1 if k == backing else k
    e_top = [remap(k) for k in e_top]
    e_bot = [remap(k) for k in e_bot]
    plate_node = [remap(k) for k in plate_node]
    nfree = node_count - 1 if backing is not None else 0

    reach = np.full(len(plates), FAR)
    if plate_reach_m is not None:
        reach[:] = plate_reach_m
    e_par = np.array([f.material.params() for f in foams]).reshape(len(foams), kernels.N_FOAM_PAR)
    e_h = np.array([f.thickness_m for f in foams], dtype=float)
    pl_D = np.array([p.bending_stiffness for p in plates], dtype=float)
    l_kind_a = np.array(l_kind, dtype=np.int64)
    l_ref_a = np.array(l_ref, dtype=np.int64)

    # rest-state spreading radii -> participating masses
    a_top = np.zeros(len(foams))
    a_plate = np.zeros(max(len(plates), 1))
    kernels.spread(np.zeros(len(foams)), 0.0, ball_radius_m, l_kind_a, l_ref_a, e_par, e_h,
                   np.ones(len(foams)), pl_D, reach, k_nts, a_top, a_plate)
    node_mass = np.zeros(nfree)
    backing_mass = 0.0

    def add(node, m):
        nonlocal backing_mass
        if node < 0:
            backing_mass += m
        else:
            node_mass[node] += m

    for j, f in enumerate(foams):
        m = f.areal_mass_kg_m2 * math.pi * max(a_top[j], ball_radius_m) ** 2
        add(e_top[j], 0.5 * m)
        add(e_bot[j], 0.5 * m)
    for p, (pl, node) in enumerate(zip(plates, plate_node)):
        add(node, pl.areal_mass_kg_m2 * math.pi * max(a_plate[p], ball_radius_m) ** 2)

    return ChainArrays(
        node_mass=np.maximum(node_mass, 1e-4), contact_node=0 if nfree > 0 else -1,
        e_top=np.array(e_top, dtype=np.int64), e_bot=np.array(e_bot, dtype=np.int64),
        e_par=e_par, e_h=e_h, l_kind=l_kind_a, l_ref=l_ref_a, pl_D=pl_D, pl_reach=reach,
        backing_mass=backing_mass, foams=tuple(foams),
    )


@dataclass
class BatchResult:
    t: np.ndarray
    force_raw_N: np.ndarray        # (N, n_t) gain applied, unfiltered
    force_N: np.ndarray            # (N, n_t) CFC-filtered
    load_cell_share: np.ndarray    # (N, n_t)
    peak_N: np.ndarray             # (N,)
    max_ball_compression_m: np.ndarray
    max_skin_compression_m: np.ndarray
    max_strain: np.ndarray         # (N, n_foam)
    energies: np.ndarray           # (N, 4) input, ball+skin dissipation, pad work, residual
    chain: ChainArrays

    @property
    def peak_lbf(self) -> np.ndarray:
        return n_to_lbf(self.peak_N)

    def bottomed(self) -> np.ndarray:
        """(N, n_foam) bool: foam reached its densification strain."""
        lim = np.array([f.material.densification_strain for f in self.chain.foams])
        return self.max_strain >= lim


def simulate_batch(balls: Sequence[Projectile], nts: NTS, velocities_ms: Sequence[float],
                   temps_F: Sequence[float], stack: Stack = (),
                   plate_reach_m: Sequence[float] | None = None, rig: Rig | None = None) -> BatchResult:
    """N impacts on one local stack, one ball/velocity/temperature per sample (parallel).
    `rig` defaults to the NOCSAE thoracic surrogate; pass Rig.drop_tower() for material tests."""
    rig = rig or Rig.nts(nts)
    solver = load_nts()["solver"]
    nd = load_nd200()
    fs = solver["output_rate_hz"]
    if fs < nd.min_sample_rate_hz:
        raise ValueError(f"output rate {fs} Hz below ND200 §9.5 minimum {nd.min_sample_rate_hz} Hz")
    dt = solver["dt_s"]
    decim = int(round(1.0 / (fs * dt)))
    n_steps = int(round(solver["duration_s"] / dt))
    n = len(balls)
    if not (len(velocities_ms) == len(temps_F) == n):
        raise ValueError("balls, velocities and temperatures must have equal length")

    R_ref = float(np.mean([b.radius_m for b in balls]))
    ch = build_chain(tuple(stack), R_ref, rig.k_found, plate_reach_m)
    e_tf = np.array([[f.material.temp_factor(T) for f in ch.foams] for T in temps_F]).reshape(n, len(ch.foams))
    args = dict(
        mb=np.array([b.mass_kg for b in balls]), kc=np.array([b.contact_stiffness for b in balls]),
        nc=np.array([b.contact_exponent for b in balls]), cor=np.array([b.cor for b in balls]),
        R=np.array([b.radius_m for b in balls]), v0=np.asarray(velocities_ms, dtype=float), e_tf=e_tf,
    )

    def run(idx: np.ndarray, dt: float):
        decim = int(round(1.0 / (fs * dt)))
        if abs(decim * fs * dt - 1.0) > 1e-9:
            raise ValueError(f"dt {dt} s must divide the output period 1/{fs} s")
        return kernels.simulate_batch(
            *(args[k][idx] for k in ("mb", "kc", "nc", "cor", "R", "v0", "e_tf")),
            ch.node_mass, ch.contact_node, ch.e_top, ch.e_bot, ch.e_par, ch.e_h, ch.l_kind, ch.l_ref,
            ch.pl_D, ch.pl_reach,
            rig.m2 + ch.backing_mass, rig.m3, rig.k2, rig.c2, rig.gain, rig.k_found, rig.r_share,
            dt, int(round(solver["duration_s"] / dt)), decim,
            solver["stop_fraction"], int(round(solver["stop_hold_s"] / dt)),
        )

    # accuracy-driven step: rerun only the samples whose energy residual is too large
    dt = solver["dt_s"]
    idx = np.arange(n)
    out_f, share, maxima, max_strain, energies = run(idx, dt)
    while True:
        bad = ~(np.abs(energies[:, 3]) <= solver["max_energy_residual"] * energies[:, 0]) | (maxima[:, 2] > 0)
        if not bad.any() or dt / 2 < solver["min_dt_s"] * (1 - 1e-9):
            break
        dt /= 2
        sub = np.flatnonzero(bad)
        f2, s2, m2_, ms2, e2 = run(sub, dt)
        out_f[sub], share[sub], maxima[sub], max_strain[sub], energies[sub] = f2, s2, m2_, ms2, e2
    if maxima[:, 2].any():
        raise RuntimeError("impact integration diverged (non-finite state); reduce solver.min_dt_s")
    # A lab record has pre-trigger samples; give the phaseless filter the same quiet lead-in so its
    # end padding does not mirror the rising edge (that shifted and distorted the simulated pulse).
    # The lead-in is kept in the returned traces (t starts at -PRE_TRIGGER_S), as in a lab record.
    pre = int(round(PRE_TRIGGER_S * fs))
    out_f = np.pad(out_f, ((0, 0), (pre, 0)))
    share = np.pad(share, ((0, 0), (pre, 0)), mode="edge")
    filt = cfc_filter(out_f, fs, nd.filter_cfc)
    return BatchResult(
        t=(np.arange(out_f.shape[1]) - pre) / fs, force_raw_N=out_f, force_N=filt, load_cell_share=share,
        peak_N=filt.max(axis=1), max_ball_compression_m=maxima[:, 0], max_skin_compression_m=maxima[:, 1],
        max_strain=max_strain, energies=energies, chain=ch,
    )


@dataclass
class ImpactResult:
    t: np.ndarray                  # s
    force_raw_N: np.ndarray        # load cell, gain applied, unfiltered
    force_N: np.ndarray            # CFC-filtered (what ND200 scores)
    velocity_ms: float
    projectile: str
    peak_N: float
    energy_in_J: float
    energy_dissipated_J: float     # ball + NTS skin damping
    energy_absorbed_pad_J: float   # net work done on the stack (0 when bare)
    energy_residual_J: float       # conservation error of the integration
    max_ball_compression_m: float
    max_skin_compression_m: float
    max_pad_compression_m: float   # sum over foam layers of peak compression (upper bound)
    max_pad_strain: float          # worst foam layer
    bottomed_layers: tuple         # descriptions of foam layers that reached densification
    min_load_cell_share: float
    beyond_standard: bool
    label: str = field(default=SIM_LABEL)

    @property
    def peak_lbf(self) -> float:
        return n_to_lbf(self.peak_N)

    @property
    def bottomed_out(self) -> bool:
        return bool(self.bottomed_layers)


def _single(res: BatchResult, i: int, projectile: Projectile, v: float) -> ImpactResult:
    foams = res.chain.foams
    strain = res.max_strain[i]
    bott = res.bottomed()[i]
    return ImpactResult(
        t=res.t, force_raw_N=res.force_raw_N[i], force_N=res.force_N[i], velocity_ms=v,
        projectile=projectile.name, peak_N=float(res.peak_N[i]),
        energy_in_J=float(res.energies[i, 0]), energy_dissipated_J=float(res.energies[i, 1]),
        energy_absorbed_pad_J=float(res.energies[i, 2]), energy_residual_J=float(res.energies[i, 3]),
        max_ball_compression_m=float(res.max_ball_compression_m[i]),
        max_skin_compression_m=float(res.max_skin_compression_m[i]),
        max_pad_compression_m=float(sum(max(s, 0.0) * f.thickness_m for s, f in zip(strain, foams))),
        max_pad_strain=float(strain.max()) if len(foams) else 0.0,
        bottomed_layers=tuple(f.describe() for f, b in zip(foams, bott) if b),
        min_load_cell_share=float(res.load_cell_share[i].min()),
        beyond_standard=not projectile.nd200,
    )


def _as_stack(pad) -> Stack:
    if pad is None:
        return ()
    if isinstance(pad, (FoamLayer, PlateLayer)):
        return (pad,)
    return tuple(pad)


def simulate_impact(projectile: Projectile, nts: NTS, velocity_ms: float, pad=None,
                    temp_F: float = 72.0, plate_reach_m: Sequence[float] | None = None,
                    rig: Rig | None = None) -> ImpactResult:
    """One impact. `pad` is None (bare NTS), a single layer, or a stack (ball side first)."""
    res = simulate_batch([projectile], nts, [velocity_ms], [temp_F], _as_stack(pad), plate_reach_m, rig)
    return _single(res, 0, projectile, velocity_ms)


def simulate_bare_nts(projectile: Projectile, nts: NTS, velocity_ms: float) -> ImpactResult:
    """Projectile into the unprotected NTS (calibration / system-check configuration)."""
    return simulate_impact(projectile, nts, velocity_ms, None)
