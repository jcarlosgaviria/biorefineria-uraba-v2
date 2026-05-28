"""
╔══════════════════════════════════════════════════════════════════════╗
║   FO2 — MINIMIZAR EMISIONES NETAS DE GEI (tCO2-eq/año)              ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║   v2    : I_min y G_max escalados a biomasa real SD (1.265M Ton/año) ║
╚══════════════════════════════════════════════════════════════════════╝
CAMBIOS v1→v2:
    - I_min default: 500,000 → 50,000,000 USD/año
    - G_max default: 250 → 150,000 tCO2/año (techo de seguridad holgado)
    - G_max calculado dinámicamente desde CAP_BASE
"""

import subprocess, sys, warnings
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pulp', '-q'])
warnings.filterwarnings('ignore')

import pulp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from datos_base import (
        ESTRUCTURAS, TEC_ELEG, PRODUCTOS, ALPHA,
        Q_BASE, CAP_BASE, C_OP, PRECIO_BASE, GEI_BASE,
        EMPLEO_BASE, R, MAX_TEC, BIOMASA_TOTAL_ANUAL, PHI_BIOCHAR
    )
except ImportError:
    exec(open('datos_base.py').read())

def _g_max_escala_real():
    try:
        q_tot = sum(Q_BASE.values())
        gei_min = (CAP_BASE.get('molienda',0) * GEI_BASE.get('molienda',0.015) +
                   CAP_BASE.get('secado',0)   * GEI_BASE.get('secado',0.045) +
                   q_tot * 0.10               * GEI_BASE.get('compostaje',0.080))
        return max(gei_min * 5, 150_000)
    except Exception:
        return 150_000

I_MIN_REAL = 50_000_000
G_MAX_REAL = _g_max_escala_real()


def resolver_FO2(
    Q_e:        dict  = None,
    cap_t:      dict  = None,
    precio:     dict  = None,
    phi:        float = None,
    gei_t:      dict  = None,
    I_min:      float = None,   # ✅ v2
    G_max:      float = None,   # ✅ v2
    costo_log:  float = 0.0,
    verbose:    bool  = True,
) -> dict:
    if Q_e    is None: Q_e    = Q_BASE.copy()
    if cap_t  is None: cap_t  = CAP_BASE.copy()
    if precio is None: precio = PRECIO_BASE.copy()
    if phi    is None: phi    = PHI_BIOCHAR
    if gei_t  is None: gei_t  = GEI_BASE.copy()
    if I_min  is None: I_min  = I_MIN_REAL
    if G_max  is None: G_max  = G_MAX_REAL

    Q_total = sum(Q_e.values())

    m = pulp.LpProblem("FO2_GEI", pulp.LpMinimize)

    x = {e: {t: pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t: pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}

    biochar_prod = pulp.lpSum(
        R[e][t]['biochar'] * x[e][t]
        for e in ESTRUCTURAS for t in TEC_ELEG if R[e][t]['biochar'] > 0
    )
    emisiones_brutas = pulp.lpSum(
        gei_t[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG
    )
    secuestro = phi * biochar_prod

    ingreso_expr = pulp.lpSum(
        precio[p] * 1000 * R[e][t][p] * x[e][t]
        for e in ESTRUCTURAS for t in TEC_ELEG for p in PRODUCTOS
        if R[e][t][p] > 0
    )
    costo_expr = pulp.lpSum(C_OP[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    utilidad   = ingreso_expr - costo_expr - costo_log

    FO2 = emisiones_brutas - secuestro
    m += FO2, "Minimizar_GEI_Neto"

    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_e[e], f"R1_disp_{e}"
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= cap_t[t] * y[t], f"R2_cap_{t}"
    m += y['molienda'] == 1, "R3_molienda"
    m += y['secado']   == 1, "R3_secado"
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC, "R4_max_tec"
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total, "R5_compostaje"
    m += utilidad >= I_min, "R6_viabilidad"
    m += emisiones_brutas <= G_max, "R7_techo_seguridad"

    estado = m.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[estado] != 'Optimal':
        if verbose:
            print(f"  FO2: {pulp.LpStatus[estado]}")
            print(f"  [diagnóstico] I_min={I_min:,.0f} USD  G_max={G_max:,.0f} tCO2")
        return {'estado': pulp.LpStatus[estado], 'FO2': 999}

    fo2_val      = pulp.value(FO2)
    em_bruto_val = pulp.value(emisiones_brutas)
    sec_val      = pulp.value(secuestro)
    biochar_val  = pulp.value(biochar_prod)
    ing_val      = pulp.value(ingreso_expr)
    costo_val    = pulp.value(costo_expr)
    util_val     = pulp.value(utilidad)
    tec_act      = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]

    gei_por_tec = {t: sum(gei_t[t]*(pulp.value(x[e][t]) or 0) for e in ESTRUCTURAS)
                   for t in tec_act}
    bio_tec = {t: sum(pulp.value(x[e][t]) or 0 for e in ESTRUCTURAS) for t in tec_act}
    uso_cap = {t: (bio_tec[t]/cap_t[t]*100) if cap_t[t]>0 else 0 for t in tec_act}
    compost = sum(R[e]['compostaje']['compost_biofertilizante']*(pulp.value(x[e]['compostaje']) or 0)
                  for e in ESTRUCTURAS)
    empleo  = sum(EMPLEO_BASE[t]*(pulp.value(x[e][t]) or 0)
                  for e in ESTRUCTURAS for t in TEC_ELEG)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SOLUCIÓN FO2 — MINIMIZAR GEI NETO")
        print(f"{'='*60}")
        print(f"  Emisiones brutas:    {em_bruto_val:>12,.1f} tCO2/año")
        print(f"  Secuestro biochar:  -{sec_val:>12,.1f} tCO2/año")
        print(f"  GEI NETO:            {fo2_val:>12,.1f} tCO2/año")
        print(f"  Biochar producido:   {biochar_val:>12,.1f} ton/año")
        print(f"  Utilidad neta:  USD  {util_val:>12,.0f}/año")
        print(f"  Tecnologías activas: {tec_act}")

    return {
        'estado':         'Optimal',
        'FO2':            fo2_val,
        'GEI_neto':       fo2_val,
        'GEI_bruto':      em_bruto_val,
        'GEI_secuestro':  sec_val,
        'biochar':        biochar_val,
        'phi_usado':      phi,
        'utilidad_neta':  util_val,
        'ingreso_bruto':  ing_val,
        'costo_biorref':  costo_val,
        'tec_activas':    tec_act,
        'tec_list':       str(tec_act),
        'gei_por_tec':    gei_por_tec,
        'bio_por_tec':    bio_tec,
        'uso_cap_pct':    uso_cap,
        'compost':        compost,
        'empleo_directo': empleo,
        'Q_total':        Q_total,
        'I_min_usado':    I_min,
        'G_max_usado':    G_max,
    }


def sensibilidad_phi(n_pasos=12):
    phis = np.linspace(0.5, 3.5, n_pasos)
    registros = []
    for p in phis:
        r = resolver_FO2(phi=p, verbose=False)
        registros.append({'phi':p,'estado':r.get('estado','Infeasible'),
                          'FO2_GEI':r.get('FO2',999),
                          'GEI_bruto':r.get('GEI_bruto',0),
                          'biochar_ton':r.get('biochar',0),
                          'util_kUSD':r.get('utilidad_neta',0)/1e3})
    return pd.DataFrame(registros)


def sensibilidad_ingreso_minimo(n_pasos=10):
    """Rango de I_min a escala real."""
    i_min_vals = np.linspace(I_MIN_REAL * 0.1, I_MIN_REAL * 3.0, n_pasos)
    registros = []
    for im in i_min_vals:
        r = resolver_FO2(I_min=im, verbose=False)
        registros.append({'I_min_kUSD':im/1e3,'estado':r.get('estado','Infeasible'),
                          'FO2_GEI':r.get('FO2',999),
                          'util_kUSD':r.get('utilidad_neta',0)/1e3})
    return pd.DataFrame(registros)


def _es_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def correr_FO2_completo():
    print(f"\n{'='*60}")
    print("  FO2 — MINIMIZAR EMISIONES NETAS DE GEI")
    print("  Biorrefinería Urabá — Escala Real SD")
    print(f"{'='*60}")
    print(f"  I_min = USD {I_MIN_REAL:,.0f}/año")
    print(f"  G_max = {G_MAX_REAL:,.0f} tCO2/año")

    print("\n  [1/3] Resolviendo modelo base...")
    res_base = resolver_FO2(verbose=True)
    if res_base.get('estado') != 'Optimal':
        print("  ERROR: modelo base infactible.")
        return

    print("\n  [2/3] Sensibilidad factor φ...")
    df_phi = sensibilidad_phi(n_pasos=10)

    print("\n  [3/3] Sensibilidad ingreso mínimo...")
    df_imin = sensibilidad_ingreso_minimo(n_pasos=8)

    print(f"\n{'='*60}")
    print(f"  FO2 = {res_base['GEI_neto']:,.1f} tCO2/año")
    print(f"  Secuestro biochar = {res_base['GEI_secuestro']:,.1f} tCO2/año")
    print(f"  Tecnologías: {res_base['tec_activas']}")
    print(f"{'='*60}")
    return res_base, df_phi, df_imin


if __name__ == '__main__' or _es_notebook():
    correr_FO2_completo()
