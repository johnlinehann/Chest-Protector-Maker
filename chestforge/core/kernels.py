"""Compiled physics kernels (numba): foam law, load spreading and the impact integrator.

Single source of truth for the impact physics — Python wrappers in padding.py / impact_sim.py
call into these.

Chain (top = ball side):
    ball -[Hunt-Crossley]-> node 0 -[foam]-> node 1 ... -[foam]-> backing ==> load cell share f -> m2
                                                                           ==> body share 1-f   -> m3
    m2 -[k2, c2 load cell]-> m3 (free on linear bearing)
A plate is a rigid node; plates bonded together share a node. A plate at the bottom of the
stack is the backing itself. The backing is massless and moves at f*v2 + (1-f)*v3.

Load spreading (top -> bottom, uniform-pressure equivalent radius a):
    top foam under the ball: footprint radius of a sphere indenting by the foam compression
    foam:  a <- a + (h - d) tan(angle)
    plate: a <- sqrt(a^2 + a_eq^2),  a_eq = sqrt(8/pi) l sqrt(r(e)),  l = (D/k)^(1/4)
      interior: plate on a Winkler foundation, w0 = P / (8 k l^2) (Timoshenko & Woinowsky-Krieger,
        Theory of Plates and Shells, 2nd ed., plates on elastic foundation). The equivalent area
        P / (k w0) matches the true PEAK pressure, so the silhouette share is an upper bound.
      near a free edge (distance e = reach): Westergaard (1926, Public Roads 7(2)) edge loading
        gives w = 0.431 P / (k l^2) vs 0.125 interior (nu = 0.15) -> area ratio 0.29 at the edge;
        r(e) = 1 - 0.71 exp(-e / l) blends to the interior value (blend shape is an estimate).
      k = foundation modulus of everything below: quasi-static secant moduli of the foams below
          in series with the NTS skin.
    f = min(1, (r_silhouette / a)^2)
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

E_MAX = 0.98            # beyond this strain the foam curve is extended linearly (keeps RK4 stable)
RATE_SMOOTH = 100.0       # 1/s, width of the load/unload blend tanh(e_dot / RATE_SMOOTH)
SQRT8_PI = math.sqrt(8.0 / math.pi)
EDGE_AREA_RATIO = 0.125 / 0.431   # Westergaard interior / edge deflection

# foam parameter columns
F_E, F_A, F_M, F_B, F_N, F_C, F_REF, F_UNL, F_TAN = range(9)
N_FOAM_PAR = 9
KIND_FOAM, KIND_PLATE = 0, 1


@njit(cache=True)
def _qs_core(e, E, A, m, B, n):
    return A * (1.0 - math.exp(-(E / A) * e * (1.0 - e) ** m)) + B * (e / (1.0 - e)) ** n


@njit(cache=True)
def foam_qs(e, E, A, m, B, n):
    """Avalle (2001) quasi-static compressive stress, Pa; linear in tension, linear beyond E_MAX."""
    if e <= 0.0:
        return E * e
    if e <= E_MAX:
        return _qs_core(e, E, A, m, B, n)
    s0 = _qs_core(E_MAX, E, A, m, B, n)
    slope = (s0 - _qs_core(E_MAX - 1e-4, E, A, m, B, n)) / 1e-4
    return s0 + slope * (e - E_MAX)


@njit(cache=True)
def foam_stress(e, e_dot, p, tf):
    """Rate-dependent stress with smooth hysteretic unloading, Pa. p = foam row, tf = temperature factor."""
    s = foam_qs(e, p[F_E], p[F_A], p[F_M], p[F_B], p[F_N]) * tf
    if e <= 0.0:
        return s
    if e_dot > p[F_REF]:
        s *= 1.0 + p[F_C] * math.log(e_dot / p[F_REF])
    r = p[F_UNL]
    return s * (0.5 * (1.0 + r) + 0.5 * (1.0 - r) * math.tanh(e_dot / RATE_SMOOTH))


@njit(cache=True)
def contact_radius(d, R):
    """Footprint radius of a sphere (radius R) indenting by d, capped at R."""
    a = min(max(d, 0.0), R)
    return math.sqrt(2.0 * R * a - a * a)


@njit(cache=True)
def spread(strain, d_ball, R, l_kind, l_ref, e_par, e_h, e_tf, pl_D, pl_reach, k_nts,
           a_top, a_plate):
    """Spreading pass. Fills a_top[j] (radius at the top of foam j) and a_plate[p]; returns final radius."""
    nl = l_kind.shape[0]
    a = 0.0
    for l in range(nl):
        if l_kind[l] == KIND_FOAM:
            j = l_ref[l]
            d = strain[j] * e_h[j]
            if l == 0:
                a = contact_radius(d, R)
            a_top[j] = a
            a = a + max(e_h[j] - max(d, 0.0), 0.0) * e_par[j, F_TAN]
        else:
            p = l_ref[l]
            if l == 0:
                a = contact_radius(d_ball, R)
            comp = 1.0 / k_nts
            for l2 in range(l + 1, nl):
                if l_kind[l2] == KIND_FOAM:
                    j2 = l_ref[l2]
                    e2 = strain[j2]
                    pr = e_par[j2]
                    if e2 > 1e-6:
                        esec = foam_qs(e2, pr[F_E], pr[F_A], pr[F_M], pr[F_B], pr[F_N]) / e2
                    else:
                        esec = pr[F_E]
                    comp += e_h[j2] / (esec * e_tf[j2])
            ell = (pl_D[p] * comp) ** 0.25
            r = 1.0 - (1.0 - EDGE_AREA_RATIO) * math.exp(-pl_reach[p] / ell)
            a_eq = SQRT8_PI * ell * math.sqrt(r)
            a = math.sqrt(a * a + a_eq * a_eq)
            a_plate[p] = a
    return a


@njit(cache=True)
def _rhs(y, dy, mb, kc, nc, alpha, R, node_mass, contact_node, e_top, e_bot, e_par, e_h, e_tf,
         l_kind, l_ref, pl_D, pl_reach, m2, m3, k2, c2, k_nts, r_sil,
         strain, force, a_top, a_plate):
    nfree = node_mass.shape[0]
    ne = e_h.shape[0]
    base = 2 + 2 * nfree
    xb = y[base + 4]
    v2, v3 = y[base + 1], y[base + 3]
    xc = xb if contact_node < 0 else y[2 + 2 * contact_node]
    d_ball = y[0] - xc

    for j in range(ne):
        xt = y[2 + 2 * e_top[j]]
        xbm = xb if e_bot[j] < 0 else y[2 + 2 * e_bot[j]]
        strain[j] = (xt - xbm) / e_h[j]
    a = spread(strain, d_ball, R, l_kind, l_ref, e_par, e_h, e_tf, pl_D, pl_reach, k_nts, a_top, a_plate)
    f = 1.0 if a <= r_sil else (r_sil / a) ** 2
    vb = f * v2 + (1.0 - f) * v3

    work = 0.0
    for k in range(base):
        dy[k] = 0.0
    for j in range(ne):
        vt = y[3 + 2 * e_top[j]]
        vbm = vb if e_bot[j] < 0 else y[3 + 2 * e_bot[j]]
        rate = (vt - vbm) / e_h[j]
        at = a_top[j]
        if strain[j] <= 0.0 and at < R:
            at = R                                     # bonded face in tension
        force[j] = foam_stress(strain[j], rate, e_par[j], e_tf[j]) * math.pi * at * at
        work += force[j] * (vt - vbm)

    vc = vb if contact_node < 0 else y[3 + 2 * contact_node]
    fc = 0.0
    e_ball = 0.0
    if d_ball > 0.0:
        elastic = kc * d_ball ** nc
        fc = max(elastic * (1.0 + alpha * (y[1] - vc)), 0.0)
        e_ball = (fc - elastic) * (y[1] - vc)

    f_back = 0.0
    if contact_node < 0:
        f_back += fc
    else:
        dy[3 + 2 * contact_node] += fc
    for j in range(ne):
        dy[3 + 2 * e_top[j]] -= force[j]
        if e_bot[j] < 0:
            f_back += force[j]
        else:
            dy[3 + 2 * e_bot[j]] += force[j]
    for k in range(nfree):
        dy[2 + 2 * k] = y[3 + 2 * k]
        dy[3 + 2 * k] /= node_mass[k]

    fs = k2 * (y[base] - y[base + 2]) + c2 * (v2 - v3)
    dy[0] = y[1]
    dy[1] = -fc / mb
    dy[base] = v2
    dy[base + 1] = (f * f_back - fs) / m2
    dy[base + 2] = v3
    dy[base + 3] = (fs + (1.0 - f) * f_back) / m3
    dy[base + 4] = vb
    dy[base + 5] = e_ball
    dy[base + 6] = c2 * (v2 - v3) ** 2
    dy[base + 7] = work
    return fs, f


@njit(cache=True, parallel=True, nogil=True)
def simulate_batch(mb, kc, nc, cor, R, v0, e_tf,
                   node_mass, contact_node, e_top, e_bot, e_par, e_h, l_kind, l_ref, pl_D, pl_reach,
                   m2, m3, k2, c2, gain, k_nts, r_sil, dt, n_steps, decim, stop_frac, hold_steps):
    """Integrate N impacts (one per sample) with fixed-step RK4.

    Per-sample arrays: mb, kc, nc, cor, R, v0 (N,), e_tf (N, ne) temperature factors.
    Early stop: once the ball has left, the load-cell force is below stop_frac x its maximum and
    hold_steps have passed since that maximum, the rest of the trace is left at zero.
    Returns force (N, n_out) [gain applied, unfiltered], load-cell share (N, n_out),
    maxima (N, 4) [ball compression, skin compression, NaN flag, stop time s], max strain (N, ne),
    energies (N, 4) [input, ball+skin dissipation, pad work, residual].
    """
    N = mb.shape[0]
    nfree = node_mass.shape[0]
    ne = e_h.shape[0]
    npl = pl_D.shape[0]
    nstate = 2 + 2 * nfree + 8
    n_out = n_steps // decim + 1
    out_f = np.zeros((N, n_out))
    out_share = np.zeros((N, n_out))
    maxima = np.zeros((N, 4))
    max_strain = np.zeros((N, ne))
    energies = np.zeros((N, 4))
    base = 2 + 2 * nfree

    for s in prange(N):
        y = np.zeros(nstate)
        y[1] = v0[s]
        k1 = np.zeros(nstate); k2_ = np.zeros(nstate); k3 = np.zeros(nstate); k4 = np.zeros(nstate)
        tmp = np.zeros(nstate)
        strain = np.zeros(ne); force = np.zeros(ne); a_top = np.zeros(ne); a_plate = np.zeros(max(npl, 1))
        alpha = 3.0 * (1.0 - cor[s] ** 2) / (4.0 * max(abs(v0[s]), 1e-6))
        tf = e_tf[s]
        mx_ball = 0.0
        mx_skin = 0.0
        f_max = 0.0
        step_at_max = 0
        last_out = 0
        stop_step = n_steps
        for step in range(n_steps + 1):
            fs, f = _rhs(y, k1, mb[s], kc[s], nc[s], alpha, R[s], node_mass, contact_node, e_top, e_bot,
                         e_par, e_h, tf, l_kind, l_ref, pl_D, pl_reach, m2, m3, k2, c2, k_nts, r_sil,
                         strain, force, a_top, a_plate)
            for j in range(ne):
                if strain[j] > max_strain[s, j]:
                    max_strain[s, j] = strain[j]
            xc = y[base + 4] if contact_node < 0 else y[2 + 2 * contact_node]
            mx_ball = max(mx_ball, y[0] - xc)
            mx_skin = max(mx_skin, y[base] - y[base + 2])
            if step % decim == 0:
                out_f[s, step // decim] = gain * fs
                out_share[s, step // decim] = f
                last_out = step // decim
            if abs(fs) > f_max:
                f_max = abs(fs)
                step_at_max = step
            if step == n_steps:
                break
            if (y[0] - xc <= 0.0 and step - step_at_max > hold_steps and abs(fs) < stop_frac * f_max
                    and step % decim == 0):
                stop_step = step
                break
            for i in range(nstate):
                tmp[i] = y[i] + 0.5 * dt * k1[i]
            _rhs(tmp, k2_, mb[s], kc[s], nc[s], alpha, R[s], node_mass, contact_node, e_top, e_bot, e_par,
                 e_h, tf, l_kind, l_ref, pl_D, pl_reach, m2, m3, k2, c2, k_nts, r_sil, strain, force, a_top, a_plate)
            for i in range(nstate):
                tmp[i] = y[i] + 0.5 * dt * k2_[i]
            _rhs(tmp, k3, mb[s], kc[s], nc[s], alpha, R[s], node_mass, contact_node, e_top, e_bot, e_par,
                 e_h, tf, l_kind, l_ref, pl_D, pl_reach, m2, m3, k2, c2, k_nts, r_sil, strain, force, a_top, a_plate)
            for i in range(nstate):
                tmp[i] = y[i] + dt * k3[i]
            _rhs(tmp, k4, mb[s], kc[s], nc[s], alpha, R[s], node_mass, contact_node, e_top, e_bot, e_par,
                 e_h, tf, l_kind, l_ref, pl_D, pl_reach, m2, m3, k2, c2, k_nts, r_sil, strain, force, a_top, a_plate)
            for i in range(nstate):
                y[i] += dt / 6.0 * (k1[i] + 2.0 * k2_[i] + 2.0 * k3[i] + k4[i])

        ke = 0.5 * mb[s] * y[1] ** 2 + 0.5 * m2 * y[base + 1] ** 2 + 0.5 * m3 * y[base + 3] ** 2
        for k in range(nfree):
            ke += 0.5 * node_mass[k] * y[3 + 2 * k] ** 2
        xc = y[base + 4] if contact_node < 0 else y[2 + 2 * contact_node]
        db = max(y[0] - xc, 0.0)
        pe = 0.5 * k2 * (y[base] - y[base + 2]) ** 2 + kc[s] * db ** (nc[s] + 1.0) / (nc[s] + 1.0)
        e_in = 0.5 * mb[s] * v0[s] ** 2
        energies[s, 0] = e_in
        energies[s, 1] = y[base + 5] + y[base + 6]
        energies[s, 2] = y[base + 7]
        energies[s, 3] = e_in - ke - pe - y[base + 5] - y[base + 6] - y[base + 7]
        maxima[s, 0] = mx_ball
        maxima[s, 1] = mx_skin
        maxima[s, 2] = 0.0 if np.isfinite(y).all() else 1.0
        maxima[s, 3] = stop_step * dt
        for k in range(last_out + 1, n_out):
            out_share[s, k] = out_share[s, last_out]
    return out_f, out_share, maxima, max_strain, energies
