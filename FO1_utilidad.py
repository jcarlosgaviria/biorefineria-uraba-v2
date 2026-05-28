"""
╔══════════════════════════════════════════════════════════════════════╗
║   FO1 — MAXIMIZAR UTILIDAD NETA (USD/año)                            ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║   v2    : I_min y G_max escalados a biomasa real SD (1.265M Ton/año) ║
╚══════════════════════════════════════════════════════════════════════╝
CAMBIOS v1→v2:
    - I_min default: 500,000 → 50,000,000 USD/año (escala real)
    - G_max default: 250 → 150,000 tCO2/año (escala real)
    - G_max se calcula dinámicamente desde BIOMASA_TOTAL_ANUAL si está disponible
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

# ── Parámetros escalados a biomasa real ──────────────────────────────
# G_max: techo GEI calculado dinámicamente según escala
# Referencia: molienda+secado+compostaje obligatorios generan ~31k tCO2/año
# a escala real. Se usa 150,000 como techo holgado (no restrictivo inicialmente).
def _g_max_escala_real():
    """Calcula techo GEI proporcional a la biomasa real del SD."""
    try:
        # Emisiones mínimas obligatorias (molienda + secado + 10% compostaje)
        q_tot = sum(Q_BASE.values())
        gei_min = (CAP_BASE.get('molienda',0) * GEI_BASE.get('molienda',0.015) +
                   CAP_BASE.get('secado',0)   * GEI_BASE.get('secado',0.045) +
                   q_tot * 0.10               * GEI_BASE.get('compostaje',0.080))
        # Techo = 5× las emisiones mínimas obligatorias (holgado, no restrictivo)
        return max(gei_min * 5, 150_000)
    except Exception:
        return 150_000

I_MIN_REAL = 50_000_000    # USD/año — escala real (~5% del ingreso potencial)
G_MAX_REAL = _g_max_escala_real()


def resolver_FO1(
    Q_e:        dict  = None,
    cap_t:      dict  = None,
    precio:     dict  = None,
    I_min:      float = None,   # ✅ v2: default calculado dinámicamente
    G_max:      float = None,   # ✅ v2: default calculado dinámicamente
    costo_log:  float = 0.0,
    verbose:    bool  = True,
) -> dict:
    if Q_e    is None: Q_e    = Q_BASE.copy()
    if cap_t  is None: cap_t  = CAP_BASE.copy()
    if precio is None: precio = PRECIO_BASE.copy()
    # ✅ v2: usar defaults escalados si no se pasan explícitamente
    if I_min  is None: I_min  = I_MIN_REAL
    if G_max  is None: G_max  = G_MAX_REAL

    Q_total = sum(Q_e.values())

    m = pulp.LpProblem("FO1_Utilidad", pulp.LpMaximize)

    x = {e: {t: pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t: pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}

    ingreso_expr = pulp.lpSum(
        precio[p] * 1000 * R[e][t][p] * x[e][t]
        for e in ESTRUCTURAS for t in TEC_ELEG for p in PRODUCTOS
        if R[e][t][p] > 0
    )
    costo_expr = pulp.lpSum(C_OP[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    utilidad   = ingreso_expr - costo_expr - costo_log
    gei_expr   = pulp.lpSum(GEI_BASE[t] * x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)

    m += utilidad, "Maximizar_Utilidad_Neta"

    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_e[e], f"R1_disp_{e}"
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= cap_t[t] * y[t], f"R2_cap_{t}"
    m += y['molienda'] == 1, "R3_molienda"
    m += y['secado']   == 1, "R3_secado"
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC, "R4_max_tec"
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10 * Q_total, "R5_compostaje"
    m += utilidad >= I_min, "R6_viabilidad"
    m += gei_expr <= G_max, "R7_GEI"

    estado = m.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[estado] != 'Optimal':
        if verbose:
            print(f"  FO1: {pulp.LpStatus[estado]}")
            print(f"  [diagnóstico] I_min={I_min:,.0f} USD  G_max={G_max:,.0f} tCO2")
        return {'estado': pulp.LpStatus[estado], 'FO1': 0}

    fo1_val   = pulp.value(utilidad)
    ing_val   = pulp.value(ingreso_expr)
    costo_val = pulp.value(costo_expr)
    gei_val   = pulp.value(gei_expr)
    tec_act   = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]

    prod    = {p: sum(R[e][t][p] * (pulp.value(x[e][t]) or 0)
                      for e in ESTRUCTURAS for t in TEC_ELEG) for p in PRODUCTOS}
    bio_tec = {t: sum(pulp.value(x[e][t]) or 0 for e in ESTRUCTURAS) for t in tec_act}
    uso_cap = {t: (bio_tec[t]/cap_t[t]*100) if cap_t[t]>0 else 0 for t in tec_act}
    compost = sum(R[e]['compostaje']['compost_biofertilizante']*(pulp.value(x[e]['compostaje']) or 0)
                  for e in ESTRUCTURAS)
    biochar = sum(R[e][t]['biochar']*(pulp.value(x[e][t]) or 0)
                  for e in ESTRUCTURAS for t in TEC_ELEG)
    empleo  = sum(EMPLEO_BASE[t]*(pulp.value(x[e][t]) or 0)
                  for e in ESTRUCTURAS for t in TEC_ELEG)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SOLUCIÓN FO1 — MAXIMIZAR UTILIDAD NETA")
        print(f"{'='*60}")
        print(f"  Ingreso bruto:       USD {ing_val:>15,.0f}/año")
        print(f"  Costo biorrefinería: USD {costo_val:>15,.0f}/año")
        print(f"  UTILIDAD NETA:       USD {fo1_val:>15,.0f}/año")
        print(f"  GEI emitido:             {gei_val:>12,.1f} tCO2/año")
        print(f"  Tecnologías activas: {tec_act}")
        print(f"  I_min usado:         USD {I_min:>15,.0f}/año")
        print(f"  G_max usado:             {G_max:>12,.0f} tCO2/año")

    return {
        'estado':          'Optimal',
        'FO1':             fo1_val,
        'ingreso_bruto':   ing_val,
        'costo_biorref':   costo_val,
        'costo_logistico': costo_log,
        'utilidad_neta':   fo1_val,
        'GEI_anual':       gei_val,
        'tec_activas':     tec_act,
        'tec_list':        str(tec_act),
        'bio_por_tec':     bio_tec,
        'uso_cap_pct':     uso_cap,
        'produccion':      prod,
        'compost':         compost,
        'biochar':         biochar,
        'empleo_directo':  empleo,
        'Q_total':         Q_total,
        'precio_usado':    precio,
        'cap_usado':       cap_t,
        'I_min_usado':     I_min,
        'G_max_usado':     G_max,
    }


def sensibilidad_precios(n_pasos=9):
    prods_clave = ['extractos_bioactivos','nanocelulosa','bioplasticos',
                   'bioetanol','biocombustible']
    factores = np.linspace(0.5, 2.0, n_pasos)
    registros = []
    for prod in prods_clave:
        for f in factores:
            precio_mod = PRECIO_BASE.copy()
            precio_mod[prod] = PRECIO_BASE[prod] * f
            r = resolver_FO1(precio=precio_mod, verbose=False)
            registros.append({'producto':prod,'factor':f,
                               'estado':r.get('estado','Infeasible'),
                               'FO1_kUSD':r.get('FO1',0)/1e3})
    return pd.DataFrame(registros)


def sensibilidad_gei_techo(n_pasos=12):
    """Análisis de sensibilidad del techo GEI — escala real."""
    gei_min = G_MAX_REAL * 0.20   # 20% del techo real
    gei_max = G_MAX_REAL * 2.0    # 200% del techo real
    registros = []
    for g in np.linspace(gei_min, gei_max, n_pasos):
        r = resolver_FO1(G_max=g, verbose=False)
        registros.append({'G_max': g,
                          'estado': r.get('estado','Infeasible'),
                          'FO1_kUSD': r.get('FO1',0)/1e3,
                          'GEI_real': r.get('GEI_anual',0),
                          'holgura': g - r.get('GEI_anual',0)})
    return pd.DataFrame(registros)


def _es_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def correr_FO1_completo():
    print(f"\n{'='*60}")
    print("  FO1 — MAXIMIZAR UTILIDAD NETA")
    print("  Biorrefinería Urabá — Escala Real SD")
    print(f"{'='*60}")
    print(f"  I_min = USD {I_MIN_REAL:,.0f}/año")
    print(f"  G_max = {G_MAX_REAL:,.0f} tCO2/año")

    print("\n  [1/3] Resolviendo modelo base...")
    res_base = resolver_FO1(verbose=True)
    if res_base.get('estado') != 'Optimal':
        print("  ERROR: modelo base infactible.")
        return

    print("\n  [2/3] Sensibilidad de precios...")
    df_precio = sensibilidad_precios(n_pasos=7)

    print("\n  [3/3] Sensibilidad techo GEI...")
    df_gei = sensibilidad_gei_techo(n_pasos=10)

    print(f"\n{'='*60}")
    print(f"  FO1 = USD {res_base['FO1']/1e6:.2f}M/año")
    print(f"  GEI = {res_base['GEI_anual']:,.0f} tCO2/año")
    print(f"  Tecnologías: {res_base['tec_activas']}")
    print(f"{'='*60}")
    return res_base, df_precio, df_gei


if __name__ == '__main__' or _es_notebook():
    correr_FO1_completo()
