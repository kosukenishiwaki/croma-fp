#!/usr/bin/env python3
"""Estimate eta = E_CR / (Q_turb * t_acc) for the eta_dpp_cap model.

This mirrors the current tracer FP convention:
  - CR spectra are tracer-total dN/dp.
  - mp and me are in GeV.
  - E_CR uses kinetic energy and converts GeV to erg.
  - Q_turb is 0.5 * M_tracer * dv^3 / L, so M_tracer cancels in eta.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


GEV_ERG = 1.6e-3
G_PER_MSUN = 1.99e33
GYR_S = 3.1536e16
KPC_CM = 3.0857e21
MP_GEV = 0.938
ME_GEV = 0.511e-3
MP_G = 1.6e-24
MU_MOL = 0.59


@dataclass(frozen=True)
class MomentumGrid:
    p: list[float]
    dp: list[float]
    e_total_gev: list[float]


def make_grid(nbin: int, logp_min: float, logp_max: float, mass_gev: float) -> MomentumGrid:
    if nbin < 3:
        raise ValueError("nbin must be >= 3")
    dlogp = (logp_max - logp_min) / (nbin - 1)
    p = [10.0 ** (logp_min + dlogp * j) for j in range(nbin)]
    pm1 = 10.0 ** (logp_min - dlogp)
    pp1 = 10.0 ** (logp_max + dlogp)
    dp = []
    for j in range(nbin):
        if j == 0:
            dp.append(0.5 * (p[1] - pm1))
        elif j == nbin - 1:
            dp.append(0.5 * (pp1 - p[nbin - 2]))
        else:
            dp.append(0.5 * (p[j + 1] - p[j - 1]))
    e_total_gev = [mass_gev * math.sqrt(1.0 + pj * pj) for pj in p]
    return MomentumGrid(p=p, dp=dp, e_total_gev=e_total_gev)


def template(p: float, delta: float, p_inj_min: float, p_inj_max: float | None) -> float:
    value = p ** (-delta) * math.exp(-p_inj_min / p)
    if p_inj_max is not None and p_inj_max > 0.0:
        value *= math.exp(-p / p_inj_max)
    return value


def energy_per_gas_gram(
    grid: MomentumGrid,
    mass_gev: float,
    phi: float,
    delta: float,
    p_inj_min: float,
    p_inj_max: float | None,
    inner_skip: int,
) -> tuple[float, float, float]:
    """Return (E_CR per gas gram, avg kinetic GeV, normalization denominator)."""
    norm_den = 0.0
    for p, dp in zip(grid.p, grid.dp):
        if p > p_inj_min:
            norm_den += template(p, delta, p_inj_min, p_inj_max) * dp
    if norm_den <= 0.0:
        raise ValueError("normalization denominator is zero")

    n_thermal_per_g = 0.52 / (MU_MOL * MP_G)
    norm_per_g = phi * n_thermal_per_g / norm_den

    ecr_per_g = 0.0
    ncr_per_g_for_energy = 0.0
    j0 = max(0, inner_skip)
    j1 = len(grid.p) - max(0, inner_skip)
    for j in range(j0, j1):
        n_dp_per_g = norm_per_g * template(grid.p[j], delta, p_inj_min, p_inj_max) * grid.dp[j]
        ekin_gev = grid.e_total_gev[j] - mass_gev
        if ekin_gev > 0.0:
            ecr_per_g += ekin_gev * GEV_ERG * n_dp_per_g
            ncr_per_g_for_energy += n_dp_per_g

    avg_kin_gev = ecr_per_g / (ncr_per_g_for_energy * GEV_ERG) if ncr_per_g_for_energy > 0.0 else 0.0
    return ecr_per_g, avg_kin_gev, norm_den


def eta_from_ecr_per_g(ecr_per_g: float, dv_kms: float, l_kpc: float, tacc_gyr: float) -> float:
    dv_cms = dv_kms * 1.0e5
    l_cm = l_kpc * KPC_CM
    q_turb_per_g = 0.5 * dv_cms**3 / l_cm
    return ecr_per_g / (q_turb_per_g * tacc_gyr * GYR_S)


def parse_csv_floats(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-cre", type=float, default=1.0e-6)
    parser.add_argument("--phi-crp", type=float, default=0.0)
    parser.add_argument("--eta-cap", type=float, default=1.0e-3)
    parser.add_argument("--delta", type=float, default=2.2)
    parser.add_argument("--l-kpc", type=float, default=150.0)
    parser.add_argument("--dv-kms", default="300,500,1000")
    parser.add_argument("--tacc-gyr", default="0.1,0.3,1.0")
    parser.add_argument("--np", type=int, default=128)
    parser.add_argument("--npe", type=int, default=128)
    parser.add_argument("--pmin", type=float, default=-1.0)
    parser.add_argument("--pmax", type=float, default=8.0)
    parser.add_argument("--pemin", type=float, default=-0.5)
    parser.add_argument("--pemax", type=float, default=6.0)
    parser.add_argument("--pinjmin", type=float, default=1.0)
    parser.add_argument("--pinjmax", type=float, default=1.0e6)
    parser.add_argument("--peinjmin", type=float, default=10.0)
    parser.add_argument("--inner-skip", type=int, default=2)
    args = parser.parse_args()

    crp_grid = make_grid(args.np, args.pmin, args.pmax, MP_GEV)
    cre_grid = make_grid(args.npe, args.pemin, args.pemax, ME_GEV)

    ecrp_per_g, avg_p_gev, _ = energy_per_gas_gram(
        crp_grid, MP_GEV, args.phi_crp, args.delta, args.pinjmin, args.pinjmax, args.inner_skip
    )
    ecre_per_g, avg_e_gev, _ = energy_per_gas_gram(
        cre_grid, ME_GEV, args.phi_cre, args.delta, args.peinjmin, None, args.inner_skip
    )
    ecr_total_per_g = ecrp_per_g + ecre_per_g

    print("eta_dpp_cap estimate")
    print(f"  phi_CRe            : {args.phi_cre:.6g}")
    print(f"  phi_CRp            : {args.phi_crp:.6g}")
    print(f"  avg CRe kinetic    : {avg_e_gev:.6g} GeV")
    print(f"  avg CRp kinetic    : {avg_p_gev:.6g} GeV")
    print(f"  E_CRe / gas mass   : {ecre_per_g:.6e} erg/g")
    print(f"  E_CRp / gas mass   : {ecrp_per_g:.6e} erg/g")
    print(f"  E_CR total / mass  : {ecr_total_per_g:.6e} erg/g")
    print(f"  L_turb             : {args.l_kpc:.6g} kpc")
    print(f"  eta_cap            : {args.eta_cap:.6g}")
    print()
    print("dv_km_s  tacc_Gyr  eta         Dpp_scale_if_capped")

    for dv_kms in parse_csv_floats(args.dv_kms):
        for tacc_gyr in parse_csv_floats(args.tacc_gyr):
            eta = eta_from_ecr_per_g(ecr_total_per_g, dv_kms, args.l_kpc, tacc_gyr)
            scale = min(1.0, args.eta_cap / eta) if eta > 0.0 and args.eta_cap > 0.0 else 1.0
            print(f"{dv_kms:8.1f} {tacc_gyr:9.3f} {eta:11.4e} {scale:19.4e}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
