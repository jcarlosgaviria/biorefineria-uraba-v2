"""
╔══════════════════════════════════════════════════════════════════════╗
║   MÓDULO DE INTEGRACIÓN MULTIOBJETIVO — FRENTE DE PARETO 5D          ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║                                                                       ║
║   VERSIÓN 2 — FO5 integrada (CAPEX + Localización)                   ║
║   Frente de Pareto: 4D → 5D                                          ║
╚══════════════════════════════════════════════════════════════════════╝

EJECUTAR EN COLAB (orden obligatorio):
    exec(open('datos_base.py').read())
    exec(open('FO1_utilidad.py').read())
    exec(open('FO2_GEI.py').read())
    exec(open('FO3_empleo.py').read())
    exec(open('FO4_aprovechamiento.py').read())
    exec(open('FO5_capex.py').read())
    exec(open('multiobjetivo_pareto_v2.py').read())

CAMBIOS vs v1:
    - FO5 agregada como quinta dimensión (min CAPEX + localización)
    - Normalización extendida a 5D
    - Métodos SP, ε, Chebyshev actualizados a 5 FO
    - Filtrado de Pareto extendido a 5 columnas norm
    - Solución compromiso L2 en espacio 5D
    - Reporte y visualizaciones actualizadas
    - FO1-FO4 sin modificaciones
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
import matplotlib.cm as cm
from itertools import product as iproduct

# ── Importar datos base y FO ──────────────────────────────────────────
try:
    from datos_base import (
        ESTRUCTURAS, TEC_ELEG, PRODUCTOS, Q_BASE, CAP_BASE,
        C_OP, PRECIO_BASE, GEI_BASE, EMPLEO_BASE, R,
        MAX_TEC, BIOMASA_TOTAL_ANUAL, PHI_BIOCHAR, MU_EMPLEO
    )
    from FO1_utilidad        import resolver_FO1
    from FO2_GEI             import resolver_FO2
    from FO3_empleo          import resolver_FO3
    from FO4_aprovechamiento import resolver_FO4
    from FO5_capex           import (resolver_FO5, CAPEX_BASE, SITIOS,
                                      WACC, HORIZONTE, PRESUPUESTO,
                                      PRECIO_CARBONO, FACTOR_INCENTIVO)
except ImportError:
    pass   # En Colab se cargan con exec()


# ══════════════════════════════════════════════════════════════════════
# S1 — PUNTO UTÓPICO Y NADIR 5D
# ══════════════════════════════════════════════════════════════════════

def calcular_utopico_nadir_5D(verbose: bool = True) -> dict:
    """
    Extiende el cálculo utópico/nadir a 5 dimensiones incluyendo FO5.

    FO5 es de MINIMIZACIÓN — utópico = mínimo CAPEX alcanzable
                              nadir   = máximo CAPEX en las otras soluciones
    """
    print("\n" + "=" * 65)
    print("  CÁLCULO DEL PUNTO UTÓPICO Y NADIR — 5D")
    print("  (FO1-FO4 sin cambios + FO5 CAPEX nuevo)")
    print("=" * 65)

    # Resolver cada FO individualmente
    print("\n  Resolviendo FO1 (max utilidad)...")
    r1 = resolver_FO1(verbose=False)
    print("  Resolviendo FO2 (min GEI)...")
    r2 = resolver_FO2(verbose=False)
    print("  Resolviendo FO3 (max empleo)...")
    r3 = resolver_FO3(verbose=False)
    print("  Resolviendo FO4 (max aprovechamiento)...")
    r4 = resolver_FO4(verbose=False)
    print("  Resolviendo FO5 (min CAPEX)...")
    r5 = resolver_FO5(verbose=False)

    resultados = [r1, r2, r3, r4, r5]

    # Extractores de cada FO
    def _fo1(r): return r.get('FO1', r.get('utilidad_neta', 0))
    def _fo2(r): return r.get('FO2', r.get('GEI_neto', r.get('GEI_anual', 0)))
    def _fo3(r): return r.get('FO3', r.get('emp_directo_total',
                               r.get('empleo_directo', 0)))
    def _fo4(r): return r.get('alpha_real', r.get('FO4', 0)) * 100
    def _fo5(r): return r.get('FO5', r.get('capex_total', 0))

    extractores = [_fo1, _fo2, _fo3, _fo4, _fo5]

    # Matriz 5×5: fila i = solución que optimiza FO_i
    matriz = np.zeros((5, 5))
    for i, res in enumerate(resultados):
        if res and not res.get('error'):
            for j, ext in enumerate(extractores):
                try:
                    matriz[i, j] = ext(res)
                except Exception:
                    matriz[i, j] = 0

    # Punto utópico: diagonal
    utopico = np.diag(matriz).copy()

    # Nadir por tipo de FO
    # FO1 (max): nadir = mínimo | FO3,FO4 (max): nadir = mínimo
    # FO2 (min): nadir = máximo | FO5 (min): nadir = máximo
    nadir = np.array([
        matriz[:, 0].min(),   # FO1 max → nadir = min
        matriz[:, 1].max(),   # FO2 min → nadir = max
        matriz[:, 2].min(),   # FO3 max → nadir = min
        matriz[:, 3].min(),   # FO4 max → nadir = min
        matriz[:, 4].max(),   # FO5 min → nadir = max
    ])

    # Si FO5 no corrió bien, usar estimado
    if utopico[4] == 0:
        utopico[4] = sum(sorted(CAPEX_BASE.values(), reverse=True)[:2]) + \
                     min(s['costo_total_sitio'] for s in SITIOS.values())
        nadir[4]   = sum(sorted(CAPEX_BASE.values(), reverse=True)[:6]) + \
                     max(s['costo_total_sitio'] for s in SITIOS.values())

    if verbose:
        unidades = ['kUSD/año', 'tCO2/año', 'emp/año', '%', 'MUSD']
        escalas  = [1/1e3, 1, 1, 1, 1/1e6]
        print(f"\n  {'FO':<10} {'Utópico':>14} {'Nadir':>14}  Unidad")
        print("  " + "-" * 55)
        for i in range(5):
            u = utopico[i] * escalas[i]
            n = nadir[i]   * escalas[i]
            print(f"  FO{i+1:<7} {u:>14.2f} {n:>14.2f}  {unidades[i]}")

    return {
        'utopico': utopico,
        'nadir':   nadir,
        'matriz':  matriz,
        'r1':r1, 'r2':r2, 'r3':r3, 'r4':r4, 'r5':r5,
    }


# ══════════════════════════════════════════════════════════════════════
# S2 — NORMALIZACIÓN MIN-MAX 5D
# ══════════════════════════════════════════════════════════════════════

def normalizar_5D(fo_vals: np.ndarray,
                  utopico: np.ndarray,
                  nadir:   np.ndarray) -> np.ndarray:
    """
    Normalización min-max para 5 FO.
    Resultado: valor 0 = utópico (mejor), 1 = nadir (peor).

    FO1,FO3,FO4 (max): norm = (utopico - f) / rango
    FO2,FO5     (min): norm = (f - utopico) / rango
    """
    norm  = np.zeros(5)
    rango = np.abs(utopico - nadir) + 1e-10

    norm[0] = (utopico[0] - fo_vals[0]) / rango[0]  # FO1 max
    norm[1] = (fo_vals[1] - utopico[1]) / rango[1]  # FO2 min
    norm[2] = (utopico[2] - fo_vals[2]) / rango[2]  # FO3 max
    norm[3] = (utopico[3] - fo_vals[3]) / rango[3]  # FO4 max
    norm[4] = (fo_vals[4] - utopico[4]) / rango[4]  # FO5 min

    return np.clip(norm, 0, 1)


# ══════════════════════════════════════════════════════════════════════
# S3 — SUMA PONDERADA 5D
# ══════════════════════════════════════════════════════════════════════

def generar_suma_ponderada_5D(utopico: np.ndarray,
                               nadir:   np.ndarray,
                               n_pesos: int = 4,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Genera puntos del frente Pareto 5D variando pesos w=(w1..w5), Σwi=1.
    """
    print(f"\n  SUMA PONDERADA 5D — generando combinaciones...")

    niveles = np.linspace(0, 1, n_pesos)
    combis  = [(w1,w2,w3,w4,w5)
               for w1 in niveles for w2 in niveles
               for w3 in niveles for w4 in niveles
               for w5 in niveles
               if abs(w1+w2+w3+w4+w5-1.0) < 1e-9]

    print(f"  Combinaciones a evaluar: {len(combis)}")
    registros = []
    for w1,w2,w3,w4,w5 in combis:
        res = _resolver_ponderado_5D(w1,w2,w3,w4,w5, utopico, nadir)
        if res is not None:
            registros.append(res)

    df = pd.DataFrame(registros)
    print(f"  Soluciones obtenidas: {len(df)}")
    return df


def _resolver_ponderado_5D(w1, w2, w3, w4, w5,
                             utopico, nadir) -> dict:
    """Problema escalarizado 5D con pesos."""
    Q_total = sum(Q_BASE.values())
    rango   = np.abs(utopico - nadir) + 1e-10
    LISTA_S = list(SITIOS.keys())

    m = pulp.LpProblem("MOO5_ponderado", pulp.LpMinimize)

    x = {e:{t:pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t:pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}
    z = {s:pulp.LpVariable(f"z_{s}", cat='Binary') for s in LISTA_S}

    ing  = pulp.lpSum(PRECIO_BASE[p]*1000*R[e][t][p]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      for p in PRODUCTOS if R[e][t][p] > 0)
    cost = pulp.lpSum(C_OP[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    gei  = pulp.lpSum(GEI_BASE[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    emp  = pulp.lpSum(EMPLEO_BASE[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    bio  = pulp.lpSum(x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    bch  = pulp.lpSum(R[e][t]['biochar']*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      if R[e][t]['biochar'] > 0)
    util = ing - cost
    fo2e = gei - PHI_BIOCHAR*bch

    # FO5: CAPEX tecnologías + costo sitio
    capex_t = pulp.lpSum(CAPEX_BASE.get(t,0)*y[t] for t in TEC_ELEG)
    cost_s  = pulp.lpSum(SITIOS[s]['costo_total_sitio']*z[s]
                          for s in LISTA_S)
    fo5e    = capex_t + cost_s

    # Función objetivo escalarizada
    obj = (w1*(utopico[0] - util)            /rango[0] +
           w2*(fo2e - utopico[1])             /rango[1] +
           w3*(utopico[2] - emp)              /rango[2] +
           w4*(utopico[3] - bio/Q_total*100)  /rango[3] +
           w5*(fo5e - utopico[4])             /rango[4])
    m += obj

    # Restricciones R1-R7 (base)
    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_BASE[e]
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= CAP_BASE[t]*y[t]
    m += y['molienda'] == 1
    m += y['secado']   == 1
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total
    m += util >= 50_000
    # R8: presupuesto CAPEX
    m += fo5e <= PRESUPUESTO * FACTOR_INCENTIVO
    # R9: exactamente un sitio
    m += pulp.lpSum(z[s] for s in LISTA_S) == 1

    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
    if pulp.LpStatus[m.status] != 'Optimal':
        return None

    fo1_v = pulp.value(util)
    fo2_v = pulp.value(fo2e)
    fo3_v = pulp.value(emp)
    fo4_v = min(pulp.value(bio)/Q_total*100, 100)
    fo5_v = pulp.value(fo5e)
    tec   = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]
    sit   = [s for s in LISTA_S if (pulp.value(z[s]) or 0) > 0.5]
    sitio = sit[0] if sit else 'N/A'

    fo_vals = np.array([fo1_v, fo2_v, fo3_v, fo4_v, fo5_v])
    fo_norm = normalizar_5D(fo_vals, utopico, nadir)

    return {
        'metodo':      'suma_ponderada',
        'w1':w1,'w2':w2,'w3':w3,'w4':w4,'w5':w5,
        'FO1_kUSD':  round(fo1_v/1e3, 2),
        'FO2_tCO2':  round(fo2_v, 3),
        'FO3_emp':   round(fo3_v, 3),
        'FO4_pct':   round(fo4_v, 2),
        'FO5_MUSD':  round(fo5_v/1e6, 3),
        'FO1_norm':  round(fo_norm[0], 4),
        'FO2_norm':  round(fo_norm[1], 4),
        'FO3_norm':  round(fo_norm[2], 4),
        'FO4_norm':  round(fo_norm[3], 4),
        'FO5_norm':  round(fo_norm[4], 4),
        'dist_utopico': round(float(np.linalg.norm(fo_norm)), 4),
        'n_tec':     len(tec),
        'tec_list':  str(tec),
        'sitio':     sitio,
    }


# ══════════════════════════════════════════════════════════════════════
# S4 — ε-RESTRICCIÓN 5D
# ══════════════════════════════════════════════════════════════════════

def generar_epsilon_5D(utopico: np.ndarray,
                        nadir:   np.ndarray,
                        n_eps:   int = 3) -> pd.DataFrame:
    """Método ε-restricción extendido a 5 FO."""
    print(f"\n  ε-RESTRICCIÓN 5D — explorando regiones no convexas...")
    registros = []

    for fo_p in [1, 2, 3, 4, 5]:
        epsilons = {}
        for fo_s in [1,2,3,4,5]:
            if fo_s == fo_p: continue
            epsilons[fo_s] = np.linspace(
                nadir[fo_s-1], utopico[fo_s-1], n_eps+2)[1:-1]

        fo_secs = [f for f in [1,2,3,4,5] if f != fo_p]
        for eps_vals in iproduct(*[epsilons[f] for f in fo_secs]):
            eps_d = dict(zip(fo_secs, eps_vals))
            res = _resolver_epsilon_5D(fo_p, eps_d, utopico, nadir)
            if res is not None:
                registros.append(res)

    df = pd.DataFrame(registros) if registros else pd.DataFrame()
    print(f"  Soluciones ε-restricción 5D: {len(df)}")
    return df


def _resolver_epsilon_5D(fo_p, eps_d, utopico, nadir):
    Q_total = sum(Q_BASE.values())
    LISTA_S = list(SITIOS.keys())
    fo_max  = {1, 3, 4}
    fo_min  = {2, 5}

    sentido = pulp.LpMinimize if fo_p in fo_min else pulp.LpMaximize
    m = pulp.LpProblem(f"MOO5_eps_FO{fo_p}", sentido)

    x = {e:{t:pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t:pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}
    z = {s:pulp.LpVariable(f"z_{s}", cat='Binary') for s in LISTA_S}

    ing  = pulp.lpSum(PRECIO_BASE[p]*1000*R[e][t][p]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      for p in PRODUCTOS if R[e][t][p] > 0)
    cost = pulp.lpSum(C_OP[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    gei  = pulp.lpSum(GEI_BASE[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    emp  = pulp.lpSum(EMPLEO_BASE[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    bio  = pulp.lpSum(x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    bch  = pulp.lpSum(R[e][t]['biochar']*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      if R[e][t]['biochar'] > 0)
    util = ing - cost
    fo2e = gei - PHI_BIOCHAR*bch
    fo5e = (pulp.lpSum(CAPEX_BASE.get(t,0)*y[t] for t in TEC_ELEG) +
            pulp.lpSum(SITIOS[s]['costo_total_sitio']*z[s] for s in LISTA_S))

    fo_exprs = {1:util, 2:fo2e, 3:emp, 4:bio, 5:fo5e}
    m += fo_exprs[fo_p]

    for fo_s, eps_v in eps_d.items():
        if fo_s in fo_max:
            m += fo_exprs[fo_s] >= eps_v
        else:
            m += fo_exprs[fo_s] <= eps_v

    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_BASE[e]
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= CAP_BASE[t]*y[t]
    m += y['molienda'] == 1
    m += y['secado']   == 1
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total
    m += util >= 50_000
    m += fo5e <= PRESUPUESTO * FACTOR_INCENTIVO
    m += pulp.lpSum(z[s] for s in LISTA_S) == 1

    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
    if pulp.LpStatus[m.status] != 'Optimal':
        return None

    fo1_v = pulp.value(util)
    fo2_v = pulp.value(fo2e)
    fo3_v = pulp.value(emp)
    fo4_v = min(pulp.value(bio)/Q_total*100, 100)
    fo5_v = pulp.value(fo5e)
    tec   = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]
    sit   = [s for s in LISTA_S if (pulp.value(z[s]) or 0) > 0.5]

    fo_vals = np.array([fo1_v, fo2_v, fo3_v, fo4_v, fo5_v])
    fo_norm = normalizar_5D(fo_vals, utopico, nadir)

    return {
        'metodo':    f'eps_FO{fo_p}',
        'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,
        'FO1_kUSD':  round(fo1_v/1e3, 2),
        'FO2_tCO2':  round(fo2_v, 3),
        'FO3_emp':   round(fo3_v, 3),
        'FO4_pct':   round(fo4_v, 2),
        'FO5_MUSD':  round(fo5_v/1e6, 3),
        'FO1_norm':  round(fo_norm[0], 4),
        'FO2_norm':  round(fo_norm[1], 4),
        'FO3_norm':  round(fo_norm[2], 4),
        'FO4_norm':  round(fo_norm[3], 4),
        'FO5_norm':  round(fo_norm[4], 4),
        'dist_utopico': round(float(np.linalg.norm(fo_norm)), 4),
        'n_tec':     len(tec),
        'tec_list':  str(tec),
        'sitio':     sit[0] if sit else 'N/A',
    }


# ══════════════════════════════════════════════════════════════════════
# S5 — CHEBYSHEV AUMENTADO 5D
# ══════════════════════════════════════════════════════════════════════

def generar_chebyshev_5D(utopico: np.ndarray,
                          nadir:   np.ndarray,
                          n_pesos: int   = 3,
                          rho:     float = 0.001) -> pd.DataFrame:
    """Chebyshev aumentado extendido a 5 FO."""
    print(f"\n  CHEBYSHEV 5D — cobertura uniforme del frente...")

    niv = np.linspace(0.05, 0.75, n_pesos)
    combis = [(w1,w2,w3,w4,w5)
              for w1 in niv for w2 in niv for w3 in niv
              for w4 in niv for w5 in niv
              if abs(w1+w2+w3+w4+w5-1.0) < 1e-9]

    print(f"  Combinaciones Chebyshev 5D: {len(combis)}")
    registros = []
    for w1,w2,w3,w4,w5 in combis:
        res = _resolver_chebyshev_5D(w1,w2,w3,w4,w5, utopico, nadir, rho)
        if res is not None:
            registros.append(res)

    df = pd.DataFrame(registros) if registros else pd.DataFrame()
    print(f"  Soluciones Chebyshev 5D: {len(df)}")
    return df


def _resolver_chebyshev_5D(w1,w2,w3,w4,w5, utopico, nadir, rho=0.001):
    Q_total = sum(Q_BASE.values())
    rango   = np.abs(utopico - nadir) + 1e-10
    LISTA_S = list(SITIOS.keys())

    m = pulp.LpProblem("MOO5_cheb", pulp.LpMinimize)

    x = {e:{t:pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t:pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}
    z = {s:pulp.LpVariable(f"z_{s}", cat='Binary') for s in LISTA_S}
    lam = pulp.LpVariable("lambda", lowBound=0)

    ing  = pulp.lpSum(PRECIO_BASE[p]*1000*R[e][t][p]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      for p in PRODUCTOS if R[e][t][p] > 0)
    cost = pulp.lpSum(C_OP[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    gei  = pulp.lpSum(GEI_BASE[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    emp  = pulp.lpSum(EMPLEO_BASE[t]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG)
    bio  = pulp.lpSum(x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    bch  = pulp.lpSum(R[e][t]['biochar']*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      if R[e][t]['biochar'] > 0)
    util = ing - cost
    fo2e = gei - PHI_BIOCHAR*bch
    fo5e = (pulp.lpSum(CAPEX_BASE.get(t,0)*y[t] for t in TEC_ELEG) +
            pulp.lpSum(SITIOS[s]['costo_total_sitio']*z[s] for s in LISTA_S))

    # Términos normalizados
    f1n = (utopico[0] - util)           /rango[0]
    f2n = (fo2e - utopico[1])           /rango[1]
    f3n = (utopico[2] - emp)            /rango[2]
    f4n = (utopico[3] - bio/Q_total*100)/rango[3]
    f5n = (fo5e - utopico[4])           /rango[4]

    m += lam + rho*(w1*f1n+w2*f2n+w3*f3n+w4*f4n+w5*f5n)
    m += lam >= w1*f1n, "cheb1"
    m += lam >= w2*f2n, "cheb2"
    m += lam >= w3*f3n, "cheb3"
    m += lam >= w4*f4n, "cheb4"
    m += lam >= w5*f5n, "cheb5"

    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_BASE[e]
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= CAP_BASE[t]*y[t]
    m += y['molienda'] == 1
    m += y['secado']   == 1
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total
    m += util >= 50_000
    m += fo5e <= PRESUPUESTO * FACTOR_INCENTIVO
    m += pulp.lpSum(z[s] for s in LISTA_S) == 1

    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
    if pulp.LpStatus[m.status] != 'Optimal':
        return None

    fo1_v = pulp.value(util)
    fo2_v = pulp.value(fo2e)
    fo3_v = pulp.value(emp)
    fo4_v = min(pulp.value(bio)/Q_total*100, 100)
    fo5_v = pulp.value(fo5e)
    tec   = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]
    sit   = [s for s in LISTA_S if (pulp.value(z[s]) or 0) > 0.5]

    fo_vals = np.array([fo1_v, fo2_v, fo3_v, fo4_v, fo5_v])
    fo_norm = normalizar_5D(fo_vals, utopico, nadir)

    return {
        'metodo':    'chebyshev',
        'w1':w1,'w2':w2,'w3':w3,'w4':w4,'w5':w5,
        'FO1_kUSD':  round(fo1_v/1e3, 2),
        'FO2_tCO2':  round(fo2_v, 3),
        'FO3_emp':   round(fo3_v, 3),
        'FO4_pct':   round(fo4_v, 2),
        'FO5_MUSD':  round(fo5_v/1e6, 3),
        'FO1_norm':  round(fo_norm[0], 4),
        'FO2_norm':  round(fo_norm[1], 4),
        'FO3_norm':  round(fo_norm[2], 4),
        'FO4_norm':  round(fo_norm[3], 4),
        'FO5_norm':  round(fo_norm[4], 4),
        'dist_utopico': round(float(np.linalg.norm(fo_norm)), 4),
        'n_tec':     len(tec),
        'tec_list':  str(tec),
        'sitio':     sit[0] if sit else 'N/A',
    }


# ══════════════════════════════════════════════════════════════════════
# S6 — FILTRADO PARETO 5D
# ══════════════════════════════════════════════════════════════════════

def filtrar_pareto_5D(df_todos: pd.DataFrame,
                       verbose:  bool = True) -> pd.DataFrame:
    """Filtra soluciones no dominadas en el espacio 5D."""
    cols_norm = ['FO1_norm','FO2_norm','FO3_norm','FO4_norm','FO5_norm']
    # Verificar columnas disponibles
    cols_ok = [c for c in cols_norm if c in df_todos.columns]
    if len(cols_ok) < 5:
        print("  Advertencia: faltan columnas FO5_norm — usando filtro 4D")
        cols_ok = [c for c in cols_norm[:4] if c in df_todos.columns]

    vals = df_todos[cols_ok].values
    n    = len(vals)
    no_dom = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j: continue
            if (np.all(vals[j] <= vals[i]) and
                    np.any(vals[j] < vals[i])):
                no_dom[i] = False
                break

    df_pareto = df_todos[no_dom].copy()
    if verbose:
        print(f"\n  FILTRADO PARETO 5D:")
        print(f"  Soluciones evaluadas:    {n}")
        print(f"  Soluciones no dominadas: {len(df_pareto)}")
        print(f"  Filtradas (dominadas):   {n - len(df_pareto)}")

    return df_pareto.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# S7 — ANÁLISIS DE CONFLICTOS 5D
# ══════════════════════════════════════════════════════════════════════

def analisis_conflictos_5D(df_pareto: pd.DataFrame) -> pd.DataFrame:
    """Correlaciones entre los 5 objetivos."""
    cols    = ['FO1_kUSD','FO2_tCO2','FO3_emp','FO4_pct','FO5_MUSD']
    nombres = ['FO1 Utilidad','FO2 GEI','FO3 Empleo',
               'FO4 Aprovech.','FO5 CAPEX']
    cols_ok = [c for c in cols if c in df_pareto.columns]
    corr    = df_pareto[cols_ok].corr()

    print("\n  CORRELACIONES ENTRE 5 OBJETIVOS (frente Pareto 5D)")
    print("=" * 65)
    print(corr.round(3).to_string())
    print("=" * 65)

    for i in range(len(cols_ok)):
        for j in range(i+1, len(cols_ok)):
            r    = corr.iloc[i, j]
            tipo = ("CONFLICTO FUERTE"   if r < -0.6 else
                    "Conflicto moderado" if r < -0.3 else
                    "Complementarios"   if r > 0.3  else
                    "Independientes")
            print(f"  {cols_ok[i][:12]} ↔ {cols_ok[j][:12]}: "
                  f"r={r:+.3f}  [{tipo}]")
    return corr


# ══════════════════════════════════════════════════════════════════════
# S8 — SOLUCIONES DE COMPROMISO 5D
# ══════════════════════════════════════════════════════════════════════

def soluciones_compromiso_5D(df_pareto: pd.DataFrame,
                               verbose:   bool = True) -> dict:
    """Soluciones de compromiso en espacio 5D."""
    cols_norm = [c for c in ['FO1_norm','FO2_norm','FO3_norm',
                              'FO4_norm','FO5_norm']
                 if c in df_pareto.columns]
    vals_norm = df_pareto[cols_norm].values

    dist_L2   = np.linalg.norm(vals_norm, axis=1)
    idx_L2    = np.argmin(dist_L2)
    dist_Linf = vals_norm.max(axis=1)
    idx_Linf  = np.argmin(dist_Linf)
    var_fo    = vals_norm.var(axis=1)
    idx_var   = np.argmin(var_fo)

    sol_L2   = df_pareto.iloc[idx_L2]
    sol_Linf = df_pareto.iloc[idx_Linf]
    sol_var  = df_pareto.iloc[idx_var]

    if verbose:
        print("\n  SOLUCIONES DE COMPROMISO — ESPACIO 5D")
        print("=" * 65)
        for sol, nombre, idx in [
            (sol_L2,   "Mínima dist. L2 (equilibrada)", idx_L2),
            (sol_Linf, "Mínima dist. L∞ (equitativa)",  idx_Linf),
            (sol_var,  "Mín. varianza (robusta)",        idx_var),
        ]:
            print(f"\n  ── {nombre} (idx={idx}) ──")
            print(f"  FO1 Utilidad:  USD {sol.get('FO1_kUSD',0):>8.1f}k/año")
            print(f"  FO2 GEI neto:      {sol.get('FO2_tCO2',0):>8.2f} tCO2/año")
            print(f"  FO3 Empleo:        {sol.get('FO3_emp',0):>8.2f} emp/año")
            print(f"  FO4 Aprovech.:     {sol.get('FO4_pct',0):>8.1f} %")
            print(f"  FO5 CAPEX:    USD  {sol.get('FO5_MUSD',0):>8.2f}M")
            print(f"  Sitio optimo:      {sol.get('sitio','N/A')}")
            print(f"  Dist. utópico 5D:  {sol.get('dist_utopico',0):.4f}")
            print(f"  Tecnologías:       {sol.get('tec_list','N/A')}")
        print("=" * 65)

    return {
        'L2':   (idx_L2,   sol_L2.to_dict()),
        'Linf': (idx_Linf, sol_Linf.to_dict()),
        'var':  (idx_var,  sol_var.to_dict()),
    }


# ══════════════════════════════════════════════════════════════════════
# S9 — VISUALIZACIONES 5D
# ══════════════════════════════════════════════════════════════════════

def graficar_pareto_5D(df_pareto: pd.DataFrame,
                        df_todos:  pd.DataFrame,
                        compromiso: dict,
                        utopico:   np.ndarray,
                        nadir:     np.ndarray):
    """Panel de 10 gráficas: todos los pares de FO en el frente 5D."""
    VERDE = '#1D9E75'; CORAL = '#D85A30'; AMBER = '#EF9F27'
    AZUL  = '#378ADD'; PURP  = '#534AB7'; GRIS  = '#B4B2A9'
    FONDO = '#F1EFE8'; OSC   = '#2C2C2A'

    cols_pares = [
        ('FO1_kUSD','FO2_tCO2','FO1 Utilidad (kUSD/año)','FO2 GEI (tCO2/año)'),
        ('FO1_kUSD','FO3_emp', 'FO1 Utilidad (kUSD/año)','FO3 Empleo (emp/año)'),
        ('FO1_kUSD','FO5_MUSD','FO1 Utilidad (kUSD/año)','FO5 CAPEX (MUSD)'),
        ('FO2_tCO2','FO3_emp', 'FO2 GEI (tCO2/año)',     'FO3 Empleo (emp/año)'),
        ('FO2_tCO2','FO5_MUSD','FO2 GEI (tCO2/año)',     'FO5 CAPEX (MUSD)'),
        ('FO3_emp', 'FO4_pct', 'FO3 Empleo (emp/año)',   'FO4 Aprovech. (%)'),
        ('FO3_emp', 'FO5_MUSD','FO3 Empleo (emp/año)',   'FO5 CAPEX (MUSD)'),
        ('FO4_pct', 'FO5_MUSD','FO4 Aprovech. (%)',      'FO5 CAPEX (MUSD)'),
        ('FO1_kUSD','FO4_pct', 'FO1 Utilidad (kUSD/año)','FO4 Aprovech. (%)'),
        ('FO2_tCO2','FO4_pct', 'FO2 GEI (tCO2/año)',     'FO4 Aprovech. (%)'),
    ]
    # Filtrar pares con columnas disponibles
    cols_ok = [c for c in df_pareto.columns]
    pares_ok = [(cx,cy,xl,yl) for cx,cy,xl,yl in cols_pares
                if cx in cols_ok and cy in cols_ok]

    n_plots = len(pares_ok)
    ncols   = 3
    nrows   = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(18, 5*nrows), facecolor='white')
    axes_flat = axes.flatten() if n_plots > 1 else [axes]
    fig.suptitle('Frente de Pareto 5D — Biorrefinería Urabá\n'
                 'FO1·FO2·FO3·FO4·FO5 — Todos los pares',
                 fontsize=14, fontweight='bold', color=OSC)

    idx_L2   = compromiso['L2'][0]
    idx_Linf = compromiso['Linf'][0]
    idx_var  = compromiso['var'][0]

    for k, (cx,cy,xl,yl) in enumerate(pares_ok):
        ax = axes_flat[k]
        ax.set_facecolor(FONDO)

        if cx in df_todos.columns and cy in df_todos.columns:
            ax.scatter(df_todos[cx], df_todos[cy],
                       c=GRIS, s=12, alpha=0.2, zorder=1)

        for met, col in [('suma_ponderada',AZUL),
                          ('chebyshev',VERDE)]:
            sub = df_pareto[df_pareto['metodo'].str.startswith(
                met.split('_')[0])]
            if len(sub):
                ax.scatter(sub[cx], sub[cy], c=col, s=30,
                           alpha=0.75, zorder=3,
                           label=met.replace('_',' '))

        sub_eps = df_pareto[df_pareto['metodo'].str.startswith('eps')]
        if len(sub_eps):
            ax.scatter(sub_eps[cx], sub_eps[cy], c=AMBER,
                       s=30, alpha=0.75, zorder=3, label='ε-restricción')

        for idx_c, col_c, mk, lb in [
            (idx_L2,  CORAL,'*','Compromiso L2'),
            (idx_Linf,PURP, 'D','Compromiso L∞'),
            (idx_var, AMBER,'P','Mín.varianza'),
        ]:
            if idx_c < len(df_pareto):
                row = df_pareto.iloc[idx_c]
                if cx in row and cy in row:
                    ax.scatter(row[cx], row[cy], c=col_c,
                               s=180, marker=mk, zorder=6,
                               edgecolors='white', lw=1.2, label=lb)

        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(f'{cx[:4]} vs {cy[:4]}', fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for k in range(len(pares_ok), len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    ruta = '/content/Pareto_5D_frente.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Figura 5D guardada: {ruta}")


def graficar_coordenadas_paralelas_5D(df_pareto: pd.DataFrame,
                                        compromiso: dict):
    """Coordenadas paralelas en el espacio 5D."""
    FONDO = '#F1EFE8'; OSC = '#2C2C2A'
    CORAL = '#D85A30'; AMBER = '#EF9F27'; PURP = '#534AB7'

    cols   = [c for c in ['FO1_kUSD','FO2_tCO2','FO3_emp',
                            'FO4_pct','FO5_MUSD']
              if c in df_pareto.columns]
    labels = ['FO1\nUtilidad\n(kUSD/año)',
              'FO2\nGEI\n(tCO2/año)',
              'FO3\nEmpleo\n(emp/año)',
              'FO4\nAprovech.\n(%)',
              'FO5\nCAPEX\n(MUSD)'][:len(cols)]

    df_plot = df_pareto[cols].copy()
    for col in cols:
        rng = df_plot[col].max() - df_plot[col].min()
        if rng > 0:
            df_plot[col] = (df_plot[col] - df_plot[col].min()) / rng

    dists     = df_pareto['dist_utopico'].values
    norm_dist = (dists - dists.min()) / max(dists.max()-dists.min(), 1e-9)

    fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')
    ax.set_facecolor(FONDO)
    fig.suptitle('Coordenadas Paralelas — Frente Pareto 5D\n'
                 'Biorrefinería Urabá · FO1·FO2·FO3·FO4·FO5',
                 fontsize=13, fontweight='bold', color=OSC)

    cmap = cm.viridis
    for idx, row in df_plot.iterrows():
        color = cmap(1 - norm_dist[idx])
        ax.plot(range(len(cols)), row.values,
                color=color, alpha=0.35, linewidth=0.9)

    for sol_idx, col, lab, lw in [
        (compromiso['L2'][0],   CORAL, 'Compromiso L2 (equilibrada)', 3),
        (compromiso['Linf'][0], PURP,  'Compromiso L∞ (equitativa)',  2.5),
        (compromiso['var'][0],  AMBER, 'Mín. varianza (robusta)',      2),
    ]:
        if sol_idx < len(df_plot):
            row = df_plot.iloc[sol_idx]
            ax.plot(range(len(cols)), row.values,
                    color=col, linewidth=lw, alpha=0.95,
                    zorder=5, label=lab)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['Peor','25%','Medio','75%','Mejor'])
    ax.set_ylabel('Valor normalizado', fontsize=10)
    ax.grid(axis='x', alpha=0.4, linewidth=1.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(dists.min(), dists.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Distancia al utópico 5D', fontsize=8)

    plt.tight_layout()
    ruta = '/content/Pareto_5D_coord_paralelas.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Coordenadas paralelas 5D: {ruta}")


# ══════════════════════════════════════════════════════════════════════
# S11 — REPORTE COMPLETO 5D
# ══════════════════════════════════════════════════════════════════════

def reporte_multiobjetivo_5D(df_pareto, compromiso, utopico, nadir, corr):
    print("\n" + "=" * 65)
    print("  REPORTE COMPLETO — OPTIMIZACIÓN MULTIOBJETIVO 5D")
    print("  Biorrefinería Urabá — Frente de Pareto FO1–FO5")
    print("=" * 65)

    unid = ['kUSD/año','tCO2/año','emp/año','%','MUSD']
    esc  = [1/1e3, 1, 1, 1, 1/1e6]
    print(f"\n  ── ESPACIO OBJETIVO 5D ──")
    for i in range(5):
        u = utopico[i]*esc[i]
        n = nadir[i]*esc[i]
        print(f"  FO{i+1}  Utópico={u:>10.2f} {unid[i]}  "
              f"Nadir={n:>10.2f} {unid[i]}")

    print(f"\n  ── FRENTE DE PARETO 5D ──")
    print(f"  Soluciones no dominadas: {len(df_pareto)}")

    print(f"\n  ── SOLUCIÓN COMPROMISO L2 (5D) ──")
    sol = compromiso['L2'][1]
    print(f"  FO1 Utilidad:  USD {sol.get('FO1_kUSD',0):>8.1f}k/año")
    print(f"  FO2 GEI neto:      {sol.get('FO2_tCO2',0):>8.2f} tCO2/año")
    print(f"  FO3 Empleo:        {sol.get('FO3_emp',0):>8.2f} emp/año")
    print(f"  FO4 Aprovech.:     {sol.get('FO4_pct',0):>8.1f} %")
    print(f"  FO5 CAPEX:    USD  {sol.get('FO5_MUSD',0):>8.2f}M")
    print(f"  Sitio optimo:      {sol.get('sitio','N/A')}")
    print(f"  Dist. utópico 5D:  {sol.get('dist_utopico',0):.4f}")
    print(f"  Tecnologías:       {sol.get('tec_list','N/A')}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
# S12 — PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════

def _es_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def correr_multiobjetivo_5D(
    n_pesos_sp: int  = 4,
    n_pesos_ch: int  = 3,
    n_eps:      int  = 2,
    verbose:    bool = True,
):
    """
    Ejecuta el análisis multiobjetivo completo en 5D:
    FO1 (util) · FO2 (GEI) · FO3 (emp) · FO4 (aprov) · FO5 (CAPEX)
    """
    print("\n" + "=" * 65)
    print("  OPTIMIZACIÓN MULTIOBJETIVO 5D — FRENTE DE PARETO")
    print("  Biorrefinería Urabá — FO1·FO2·FO3·FO4·FO5")
    print("=" * 65)

    # 1. Utópico y nadir 5D
    print("\n  [1/7] Calculando punto utópico y nadir 5D...")
    un = calcular_utopico_nadir_5D(verbose=verbose)
    utopico, nadir = un['utopico'], un['nadir']

    # 2. Suma ponderada 5D
    print("\n  [2/7] Generando puntos — suma ponderada 5D...")
    df_sp = generar_suma_ponderada_5D(utopico, nadir, n_pesos=n_pesos_sp)

    # 3. ε-restricción 5D
    print("\n  [3/7] Generando puntos — ε-restricción 5D...")
    df_ep = generar_epsilon_5D(utopico, nadir, n_eps=n_eps)

    # 4. Chebyshev 5D
    print("\n  [4/7] Generando puntos — Chebyshev 5D...")
    df_ch = generar_chebyshev_5D(utopico, nadir, n_pesos=n_pesos_ch)

    # Combinar
    dfs = [df for df in [df_sp, df_ep, df_ch] if len(df) > 0]
    if not dfs:
        print("  ERROR: No se generaron soluciones. Verificar parámetros.")
        return None

    cols_dedup = ['FO1_kUSD','FO2_tCO2','FO3_emp','FO4_pct','FO5_MUSD']
    cols_dedup_ok = [c for c in cols_dedup
                     if all(c in df.columns for df in dfs)]
    df_todos = pd.concat(dfs, ignore_index=True)
    if cols_dedup_ok:
        df_todos = df_todos.drop_duplicates(subset=cols_dedup_ok)
    print(f"\n  Total soluciones únicas evaluadas: {len(df_todos)}")

    # 5. Filtrar frente Pareto 5D
    print("\n  [5/7] Filtrando frente de Pareto 5D...")
    df_pareto = filtrar_pareto_5D(df_todos, verbose=verbose)

    # 6. Conflictos
    print("\n  [6/7] Análisis de conflictos 5D...")
    corr = analisis_conflictos_5D(df_pareto)

    # 7. Compromiso
    print("\n  [7/7] Soluciones de compromiso 5D...")
    compromiso = soluciones_compromiso_5D(df_pareto, verbose=verbose)

    # Visualizaciones
    print("\n  Generando frente de Pareto 5D (10 pares)...")
    graficar_pareto_5D(df_pareto, df_todos, compromiso, utopico, nadir)

    print("\n  Generando coordenadas paralelas 5D...")
    graficar_coordenadas_paralelas_5D(df_pareto, compromiso)

    # Reporte
    reporte_multiobjetivo_5D(df_pareto, compromiso, utopico, nadir, corr)

    # Guardar CSV
    ruta_csv = '/content/frente_pareto_5D_uraba.csv'
    df_pareto.to_csv(ruta_csv, index=False)
    print(f"\n  Frente Pareto 5D guardado: {ruta_csv}")
    print(f"  ({len(df_pareto)} soluciones no dominadas)")

    return {
        'df_todos':   df_todos,
        'df_pareto':  df_pareto,
        'utopico':    utopico,
        'nadir':      nadir,
        'compromiso': compromiso,
        'corr':       corr,
    }


if __name__ == '__main__' or _es_notebook():
    resultados_5D = correr_multiobjetivo_5D(
        n_pesos_sp=4,
        n_pesos_ch=3,
        n_eps=2,
    )
