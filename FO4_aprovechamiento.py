"""
╔══════════════════════════════════════════════════════════════════════╗
║   FO4 — MAXIMIZAR APROVECHAMIENTO DE BIOMASA (%)                     ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║   v2    : I_min y G_max escalados a biomasa real SD (1.265M Ton/año) ║
╚══════════════════════════════════════════════════════════════════════╝
CAMBIOS v1→v2:
    - I_min default: 200,000 → 50,000,000 USD/año
    - G_max default: 250 → 150,000 tCO2/año
    - ETA_CADENA_DEFAULT: 0.42 → cargado desde SD_PARAMS si disponible
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

# ── Parámetros específicos de FO4 ────────────────────────────────────
# ✅ v2: eta_cadena cargado desde SD_PARAMS si está disponible
try:
    ETA_CADENA_DEFAULT = SD_PARAMS.get('eta_cadena_sd', 0.42)
except NameError:
    ETA_CADENA_DEFAULT = 0.42

TASA_BIO_HA_MES = 3.0
S_SUP_INICIAL   = 36_932.0   # ✅ v2: superficie real del SD

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


def resolver_FO4(
    Q_e:        dict  = None,
    Q_gen:      float = None,
    cap_t:      dict  = None,
    precio:     dict  = None,
    eta_cadena: float = None,
    I_min:      float = None,   # ✅ v2
    G_max:      float = None,   # ✅ v2
    costo_log:  float = 0.0,
    verbose:    bool  = True,
) -> dict:
    if Q_e        is None: Q_e        = Q_BASE.copy()
    if cap_t      is None: cap_t      = CAP_BASE.copy()
    if precio     is None: precio     = PRECIO_BASE.copy()
    if eta_cadena is None: eta_cadena = ETA_CADENA_DEFAULT
    if I_min      is None: I_min      = I_MIN_REAL
    if G_max      is None: G_max      = G_MAX_REAL

    Q_total = sum(Q_e.values())

    if Q_gen is None:
        Q_gen = Q_total / eta_cadena if eta_cadena > 0 else Q_total

    m = pulp.LpProblem("FO4_Aprovechamiento", pulp.LpMaximize)

    x = {e: {t: pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t: pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}

    FO4 = pulp.lpSum(x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    m += FO4, "Maximizar_Biomasa_Procesada"

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
            print(f"  FO4: {pulp.LpStatus[estado]}")
            print(f"  [diagnóstico] I_min={I_min:,.0f} USD  G_max={G_max:,.0f} tCO2")
        return {'estado': pulp.LpStatus[estado], 'FO4': 0,
                'alpha_real': 0, 'alpha_red': 0, 'alpha_BR': 0}

    fo4_val  = pulp.value(FO4)
    gei_val  = pulp.value(gei_expr)
    util_val = pulp.value(utilidad)
    ing_val  = pulp.value(ingreso_expr)
    tec_act  = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]

    alpha_real = min(fo4_val / Q_gen,   1.0) if Q_gen   > 0 else 0
    alpha_red  = min(Q_total / Q_gen,   1.0) if Q_gen   > 0 else 0
    alpha_BR   = min(fo4_val / Q_total, 1.0) if Q_total > 0 else 0

    perdida_red = Q_gen   - Q_total
    perdida_BR  = Q_total - fo4_val

    bio_tec = {t: sum(pulp.value(x[e][t]) or 0 for e in ESTRUCTURAS) for t in tec_act}
    uso_cap = {t: (bio_tec[t]/cap_t[t]*100) if cap_t[t]>0 else 0 for t in tec_act}
    empleo  = sum(EMPLEO_BASE[t]*(pulp.value(x[e][t]) or 0)
                  for e in ESTRUCTURAS for t in TEC_ELEG)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SOLUCIÓN FO4 — MAXIMIZAR APROVECHAMIENTO")
        print(f"{'='*60}")
        print(f"  Biomasa generada:    {Q_gen:>12,.0f} ton/año")
        print(f"  Biomasa recolectada: {Q_total:>12,.0f} ton/año")
        print(f"  Biomasa procesada:   {fo4_val:>12,.0f} ton/año")
        print(f"  α_red  (campo→acopio): {alpha_red*100:>6.1f}%")
        print(f"  α_BR   (acopio→BR):    {alpha_BR*100:>6.1f}%")
        print(f"  α_real (campo→BR):     {alpha_real*100:>6.1f}%  ← principal")
        print(f"  Pérdida en red:      {perdida_red:>12,.0f} ton/año")
        print(f"  Pérdida en BR:       {perdida_BR:>12,.0f} ton/año")
        print(f"  GEI emitido:         {gei_val:>12,.1f} tCO2/año")
        print(f"  Tecnologías activas: {tec_act}")
        print(f"  eta_cadena usado:    {eta_cadena:.4f}")

    return {
        'estado':         'Optimal',
        'FO4':            fo4_val,
        'alpha_real':     alpha_real,
        'alpha_red':      alpha_red,
        'alpha_BR':       alpha_BR,
        'perdida_red':    perdida_red,
        'perdida_BR':     perdida_BR,
        'Q_gen':          Q_gen,
        'Q_total':        Q_total,
        'GEI_anual':      gei_val,
        'utilidad_neta':  util_val,
        'ingreso_bruto':  ing_val,
        'tec_activas':    tec_act,
        'tec_list':       str(tec_act),
        'bio_por_tec':    bio_tec,
        'uso_cap_pct':    uso_cap,
        'empleo_directo': empleo,
        'eta_usado':      eta_cadena,
        'I_min_usado':    I_min,
        'G_max_usado':    G_max,
    }


def sensibilidad_eta_cadena(n_pasos=12):
    etas = np.linspace(0.20, 0.90, n_pasos)
    registros = []
    for eta in etas:
        r = resolver_FO4(eta_cadena=eta, verbose=False)
        registros.append({'eta': eta,
                          'estado': r.get('estado','Infeasible'),
                          'alpha_real': r.get('alpha_real',0)*100,
                          'FO4_ton': r.get('FO4',0)})
    return pd.DataFrame(registros)


def _es_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def correr_FO4_completo():
    print(f"\n{'='*60}")
    print("  FO4 — MAXIMIZAR APROVECHAMIENTO DE BIOMASA")
    print("  Biorrefinería Urabá — Escala Real SD")
    print(f"{'='*60}")
    print(f"  I_min      = USD {I_MIN_REAL:,.0f}/año")
    print(f"  G_max      = {G_MAX_REAL:,.0f} tCO2/año")
    print(f"  eta_cadena = {ETA_CADENA_DEFAULT:.4f}")

    print("\n  [1/2] Resolviendo modelo base...")
    res_base = resolver_FO4(verbose=True)
    if res_base.get('estado') != 'Optimal':
        print("  ERROR: modelo base infactible.")
        return

    print("\n  [2/2] Sensibilidad η_cadena...")
    df_eta = sensibilidad_eta_cadena(n_pasos=10)

    print(f"\n{'='*60}")
    print(f"  α_real = {res_base['alpha_real']*100:.1f}%")
    print(f"  α_BR   = {res_base['alpha_BR']*100:.1f}%")
    print(f"  Tecnologías: {res_base['tec_activas']}")
    print(f"{'='*60}")
    return res_base, df_eta


if __name__ == '__main__' or _es_notebook():
    correr_FO4_completo()
