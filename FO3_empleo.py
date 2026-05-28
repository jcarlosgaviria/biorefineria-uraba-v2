"""
╔══════════════════════════════════════════════════════════════════════╗
║   FO3 — MAXIMIZAR EMPLEO GENERADO (empleos directos/año)             ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║   v2    : I_min y G_max escalados a biomasa real SD (1.265M Ton/año) ║
╚══════════════════════════════════════════════════════════════════════╝
CAMBIOS v1→v2:
    - I_min default: 300,000 → 50,000,000 USD/año
    - G_max default: 200 → 150,000 tCO2/año
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
        EMPLEO_BASE, R, MAX_TEC, BIOMASA_TOTAL_ANUAL,
        PHI_BIOCHAR, MU_EMPLEO
    )
except ImportError:
    exec(open('datos_base.py').read())

# ── Parámetros específicos de FO3 ────────────────────────────────────
EMPLEO_CAMPO_POR_TON  = 0.003
EMPLEO_ACOPIO_POR_TON = 0.002
EMPLEO_ADMIN_CENTRO   = 2.0
N_CENTROS_DEFAULT     = 4

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


def resolver_FO3(
    Q_e:            dict  = None,
    cap_t:          dict  = None,
    precio:         dict  = None,
    empleo_t:       dict  = None,
    emp_campo:      float = None,
    emp_acopio:     float = None,
    n_centros:      int   = None,
    mu:             float = None,
    I_min:          float = None,   # ✅ v2
    G_max:          float = None,   # ✅ v2
    costo_log:      float = 0.0,
    verbose:        bool  = True,
) -> dict:
    if Q_e        is None: Q_e        = Q_BASE.copy()
    if cap_t      is None: cap_t      = CAP_BASE.copy()
    if precio     is None: precio     = PRECIO_BASE.copy()
    if empleo_t   is None: empleo_t   = EMPLEO_BASE.copy()
    if emp_campo  is None: emp_campo  = EMPLEO_CAMPO_POR_TON
    if emp_acopio is None: emp_acopio = EMPLEO_ACOPIO_POR_TON
    if n_centros  is None: n_centros  = N_CENTROS_DEFAULT
    if mu         is None: mu         = MU_EMPLEO
    if I_min      is None: I_min      = I_MIN_REAL
    if G_max      is None: G_max      = G_MAX_REAL

    Q_total = sum(Q_e.values())

    m = pulp.LpProblem("FO3_Empleo", pulp.LpMaximize)

    x = {e: {t: pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t: pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}

    emp_biorref_expr = pulp.lpSum(
        empleo_t[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG
    )
    biomasa_usada    = pulp.lpSum(x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    emp_campo_expr   = emp_campo * biomasa_usada
    emp_acopio_val   = emp_acopio * Q_total + n_centros * EMPLEO_ADMIN_CENTRO

    FO3 = emp_biorref_expr + emp_campo_expr
    m += FO3, "Maximizar_Empleo_Directo"

    ingreso_expr = pulp.lpSum(
        precio[p] * 1000 * R[e][t][p] * x[e][t]
        for e in ESTRUCTURAS for t in TEC_ELEG for p in PRODUCTOS
        if R[e][t][p] > 0
    )
    costo_expr = pulp.lpSum(C_OP[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    utilidad   = ingreso_expr - costo_expr - costo_log
    gei_expr   = pulp.lpSum(GEI_BASE[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)

    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_e[e], f"R1_disp_{e}"
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= cap_t[t] * y[t], f"R2_cap_{t}"
    m += y['molienda'] == 1, "R3_molienda"
    m += y['secado']   == 1, "R3_secado"
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC, "R4_max_tec"
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total, "R5_compostaje"
    m += utilidad >= I_min, "R6_viabilidad"
    m += gei_expr <= G_max, "R7_GEI"

    estado = m.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[estado] != 'Optimal':
        if verbose:
            print(f"  FO3: {pulp.LpStatus[estado]}")
            print(f"  [diagnóstico] I_min={I_min:,.0f} USD  G_max={G_max:,.0f} tCO2")
        return {'estado': pulp.LpStatus[estado], 'FO3': 0}

    fo3_val     = pulp.value(FO3)
    emp_br_val  = pulp.value(emp_biorref_expr)
    emp_cam_val = pulp.value(emp_campo_expr)
    bio_val     = pulp.value(biomasa_usada)
    gei_val     = pulp.value(gei_expr)
    util_val    = pulp.value(utilidad)
    tec_act     = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]

    emp_dir_total = fo3_val + emp_acopio_val
    emp_indirecto = emp_dir_total * mu
    emp_total     = emp_dir_total + emp_indirecto

    emp_por_tec = {t: sum(empleo_t[t]*(pulp.value(x[e][t]) or 0) for e in ESTRUCTURAS)
                   for t in tec_act}
    bio_tec = {t: sum(pulp.value(x[e][t]) or 0 for e in ESTRUCTURAS) for t in tec_act}
    uso_cap = {t: (bio_tec[t]/cap_t[t]*100) if cap_t[t]>0 else 0 for t in tec_act}
    compost = sum(R[e]['compostaje']['compost_biofertilizante']*(pulp.value(x[e]['compostaje']) or 0)
                  for e in ESTRUCTURAS)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SOLUCIÓN FO3 — MAXIMIZAR EMPLEO")
        print(f"{'='*60}")
        print(f"  Empleo biorrefinería:  {emp_br_val:>10,.1f} emp/año")
        print(f"  Empleo campo:          {emp_cam_val:>10,.1f} emp/año")
        print(f"  Empleo centros acopio: {emp_acopio_val:>10,.1f} emp/año")
        print(f"  EMPLEO DIRECTO:        {emp_dir_total:>10,.1f} emp/año")
        print(f"  Empleo indirecto (×{mu:.1f}):{emp_indirecto:>9,.1f} emp/año")
        print(f"  EMPLEO TOTAL:          {emp_total:>10,.1f} emp/año")
        print(f"  GEI emitido:           {gei_val:>10,.1f} tCO2/año")
        print(f"  Tecnologías activas:   {tec_act}")

    return {
        'estado':             'Optimal',
        'FO3':                fo3_val,
        'emp_biorref':        emp_br_val,
        'emp_campo':          emp_cam_val,
        'emp_acopio':         emp_acopio_val,
        'emp_directo_total':  emp_dir_total,
        'emp_indirecto':      emp_indirecto,
        'emp_total':          emp_total,
        'mu_usado':           mu,
        'GEI_anual':          gei_val,
        'utilidad_neta':      util_val,
        'tec_activas':        tec_act,
        'tec_list':           str(tec_act),
        'emp_por_tec':        emp_por_tec,
        'bio_por_tec':        bio_tec,
        'uso_cap_pct':        uso_cap,
        'compost':            compost,
        'Q_total':            Q_total,
        'I_min_usado':        I_min,
        'G_max_usado':        G_max,
    }


def sensibilidad_multiplicador_mu(n_pasos=13):
    mus = np.linspace(1.0, 5.0, n_pasos)
    res_base = resolver_FO3(verbose=False)
    registros = []
    for mu_val in mus:
        if res_base.get('estado') == 'Optimal':
            ed = res_base['emp_directo_total']
            ei = ed * mu_val
            registros.append({'mu': mu_val,
                               'emp_directo': ed,
                               'emp_indirecto': ei,
                               'emp_total': ed + ei})
    return pd.DataFrame(registros)


def _es_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def correr_FO3_completo():
    print(f"\n{'='*60}")
    print("  FO3 — MAXIMIZAR EMPLEO GENERADO")
    print("  Biorrefinería Urabá — Escala Real SD")
    print(f"{'='*60}")
    print(f"  I_min = USD {I_MIN_REAL:,.0f}/año")
    print(f"  G_max = {G_MAX_REAL:,.0f} tCO2/año")

    print("\n  [1/2] Resolviendo modelo base...")
    res_base = resolver_FO3(verbose=True)
    if res_base.get('estado') != 'Optimal':
        print("  ERROR: modelo base infactible.")
        return

    print("\n  [2/2] Sensibilidad multiplicador μ...")
    df_mu = sensibilidad_multiplicador_mu(n_pasos=10)

    print(f"\n{'='*60}")
    print(f"  FO3 directo = {res_base['emp_directo_total']:,.1f} emp/año")
    print(f"  FO3 total   = {res_base['emp_total']:,.1f} emp/año")
    print(f"  Tecnologías: {res_base['tec_activas']}")
    print(f"{'='*60}")
    return res_base, df_mu


if __name__ == '__main__' or _es_notebook():
    correr_FO3_completo()
