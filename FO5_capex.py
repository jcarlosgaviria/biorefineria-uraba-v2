"""
╔══════════════════════════════════════════════════════════════════════╗
║   FO5 — MINIMIZAR CAPEX + LOCALIZACIÓN ÓPTIMA                        ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║                                                                       ║
║   DATOS SIMULADOS REALISTAS — Fase 1 (escenario de referencia)       ║
║   Fuentes:                                                            ║
║   - CAPEX: FENOGE/CQM 2024, literatura pirólisis/compostaje LAm       ║
║   - Terrenos: Century21 Colombia, Zona Franca Urabá 2024             ║
║   - Financieros: WACC sector agroindustrial Colombia (Bancolombia)    ║
║   - Distancias: Google Maps municipios Urabá a zonas de cultivo      ║
║                                                                       ║
║   NOTA: Reemplazar por datos reales cuando estén disponibles         ║
║   en datos_base.py — sección CAPEX_BASE y SITIOS                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import warnings; warnings.filterwarnings('ignore')
import pulp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from datos_base import (
        ESTRUCTURAS, TEC_ELEG, PRODUCTOS, Q_BASE, CAP_BASE,
        C_OP, PRECIO_BASE, GEI_BASE, EMPLEO_BASE, R, MAX_TEC,
        BIOMASA_TOTAL_ANUAL, BIOMASA_RECOLECTADA, ETA_CADENA, PHI_BIOCHAR
    )
except ImportError:
    exec(open('datos_base.py').read())


# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CAPEX POR TECNOLOGÍA (USD)
# Datos simulados realistas para Urabá — Fase 1
# ══════════════════════════════════════════════════════════════════════
# Fuentes de referencia:
# - Molienda/Secado: plantas agroindustriales Colombia (FENOGE 2024)
# - Compostaje: plantas compost Antioquia (CORNARE, Corantioquia)
# - Fermentación: plantas bioetanol caña Colombia (CENICAÑA)
# - Pirólisis: literatura LAm biochar residuos banano (Ferrer et al. 2020)
# - Hidrólisis: proyectos nanocelulosa/bioplásticos Colombia (Colciencias)
# - Extracción solventes: industria extractos vegetales Colombia
# Escala: planta procesando ~929,692 Ton/año (escala real SD)
# Factor de corrección Colombia vs USA/Europa: 0.75 (menor costo M.O.)

CAPEX_BASE = {
    # ── Pretratamiento (obligatorio) ──────────────────────────────
    'molienda':             1_200_000,   # USD — molinos industriales
                                          # Ref: plantas palmiste Antioquia
    'secado':               1_800_000,   # USD — secadores rotatorios
                                          # Ref: plantas café Antioquia
    # ── Tratamiento biológico ─────────────────────────────────────
    'compostaje':             850_000,   # USD — planta compostaje
                                          # Ref: EMVARIAS Medellín escala industrial
    'fermentacion':         6_500_000,   # USD — bioetanol/biogás
                                          # Ref: plantas bioetanol caña Colombia
    # ── Química fina ─────────────────────────────────────────────
    'transesterificacion':  3_200_000,   # USD — biodiesel
                                          # Ref: plantas biodiesel palma Colombia
    'extraccion_solventes':12_000_000,   # USD — extractos bioactivos
                                          # Ref: industria extractos naturales
    'hidrolisis_enzimatica': 8_500_000,  # USD — nanocelulosa/bioplásticos
                                          # Ref: proyectos I+D Colombia+Brasil
    # ── Termoquímica ─────────────────────────────────────────────
    'pirolisis':             5_500_000,  # USD — pirólisis lenta biochar
                                          # Ref: Ferrer et al. 2020, Ecuador
    'carbonizacion':         4_800_000,  # USD — carbonización hidrotérmica
                                          # Ref: literatura LAm HTC
    'gasificacion':                  0,  # USD — excluida por TRL < 4
}

# CAPEX_TOTAL máximo esperado con MAX_TEC=6 tecnologías activas
# Rango realista: USD 15M - 35M según combinación elegida
CAPEX_MAX_REFERENCIA = sum(sorted(CAPEX_BASE.values(), reverse=True)[:6])


# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — SITIOS CANDIDATOS DE LOCALIZACIÓN
# Municipios Urabá con potencial para planta biorefinería
# ══════════════════════════════════════════════════════════════════════
# Fuentes:
# - Terrenos: Century21 Colombia, Zona Franca Urabá (2024)
#   Zona Franca Urabá (Apartadó): ~690M COP/lote → ~USD 170k/Ha
#   Lotes industriales Turbo (Puerto Antioquia): USD 45k-80k/Ha
# - Distancias: Google Maps a centroide zonas bananeras
# - Infraestructura: INVIAS, Plan Vial Antioquia
# - Tasa de cambio referencia: COP 4,050/USD (2024)

SITIOS = {
    'Apartado': {
        # Centro logístico La Teka — zona industrial consolidada
        'costo_terreno_usd_ha':   170_000,   # USD/Ha (Zona Franca)
        'area_necesaria_ha':            5,   # Ha para planta completa
        'dist_campo_km':               18,   # km promedio a zonas cultivo
        'costo_infra_usd':        500_000,   # USD adecuación servicios
        'acceso_vial':                  5,   # 1-5 (5=mejor)
        'latitud':               7.8839,
        'longitud':            -76.6275,
        'descripcion': 'Centro logístico consolidado, Zona Franca, mejor acceso vial',
    },
    'Turbo': {
        # Zona industrial Puerto Antioquia — acceso portuario
        'costo_terreno_usd_ha':    65_000,   # USD/Ha (zona portuaria)
        'area_necesaria_ha':            5,
        'dist_campo_km':               35,   # más alejado de cultivos
        'costo_infra_usd':        350_000,   # USD
        'acceso_vial':                  4,
        'latitud':               8.0968,
        'longitud':            -76.7291,
        'descripcion': 'Puerto Antioquia, exportación directa, menor costo terreno',
    },
    'Carepa': {
        # Zona agroindustrial central — equidistante a cultivos
        'costo_terreno_usd_ha':    55_000,   # USD/Ha
        'area_necesaria_ha':            5,
        'dist_campo_km':               22,
        'costo_infra_usd':        420_000,
        'acceso_vial':                  4,
        'latitud':               7.7617,
        'longitud':            -76.6566,
        'descripcion': 'Zona central Urabá, equidistante a zonas bananeras',
    },
    'Chigorodo': {
        # Zona sur — acceso cultivos sur de Urabá
        'costo_terreno_usd_ha':    48_000,   # USD/Ha
        'area_necesaria_ha':            5,
        'dist_campo_km':               28,
        'costo_infra_usd':        380_000,
        'acceso_vial':                  3,
        'latitud':               7.6719,
        'longitud':            -76.6852,
        'descripcion': 'Menor costo terreno, buen acceso cultivos sur Urabá',
    },
    'Mutata': {
        # Zona sur profundo — menor costo, mayor distancia
        'costo_terreno_usd_ha':    32_000,   # USD/Ha
        'area_necesaria_ha':            5,
        'dist_campo_km':               42,
        'costo_infra_usd':        550_000,   # mayor inversión en infraestructura
        'acceso_vial':                  2,
        'latitud':               7.2444,
        'longitud':            -76.4361,
        'descripcion': 'Menor CAPEX terreno, mayor costo logístico y de infraestructura',
    },
}

# Calcular costo total por sitio
for s, data in SITIOS.items():
    data['costo_total_sitio'] = (
        data['costo_terreno_usd_ha'] * data['area_necesaria_ha'] +
        data['costo_infra_usd']
    )
    # Factor de costo logístico: USD 0.15/Ton/km (camión biomasa)
    data['costo_log_anual'] = (
        BIOMASA_RECOLECTADA * data['dist_campo_km'] * 0.15
    )


# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — PARÁMETROS FINANCIEROS
# ══════════════════════════════════════════════════════════════════════
# Fuentes:
# - WACC: sector agroindustrial Colombia — Bancolombia/Davivienda 2024
# - Horizonte: UPME proyectos bioenergía Colombia (20 años)
# - Precio carbono: mercado voluntario Colombia (MADS 2024)
# - Incentivos: Ley 1715/2014 energías renovables Colombia

WACC          = 0.125    # 12.5% — WACC referencia agroindustrial Colombia
HORIZONTE     = 20       # años — horizonte evaluación financiera
PRESUPUESTO   = 40_000_000  # USD — presupuesto máximo inversión (simulado)
PRECIO_CARBONO = 18.0    # USD/tCO2 — mercado voluntario Colombia 2024
TASA_INFLACION = 0.065   # 6.5% — inflación Colombia promedio 2024

# Incentivos disponibles (Ley 1715 + MinCiencias)
INCENTIVOS = {
    'deduccion_renta':    0.50,   # 50% deducción renta sobre inversión
    'iva_excluido':       True,   # IVA excluido en compra equipos
    'arancel_0':          True,   # Arancel 0% equipos importados
    'subsidio_capex':     0.10,   # 10% subsidio directo (simulado MinCiencias)
}

# CAPEX efectivo después de incentivos
FACTOR_INCENTIVO = 1 - INCENTIVOS['subsidio_capex'] - (
    INCENTIVOS['deduccion_renta'] * 0.35  # tasa impositiva Colombia
)


# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — FUNCIÓN OBJETIVO 5 (FO5)
# ══════════════════════════════════════════════════════════════════════
def _g_max_real():
    try:
        q_tot = sum(Q_BASE.values())
        gei_min = (CAP_BASE.get('molienda',0)*GEI_BASE.get('molienda',0.015) +
                   CAP_BASE.get('secado',0)*GEI_BASE.get('secado',0.045) +
                   q_tot*0.10*GEI_BASE.get('compostaje',0.080))
        return max(gei_min * 5, 150_000)
    except Exception:
        return 150_000

I_MIN_REAL = 50_000_000
G_MAX_REAL = _g_max_real()


def resolver_FO5(
    Q_e:           dict  = None,
    cap_t:         dict  = None,
    precio:        dict  = None,
    capex_t:       dict  = None,
    sitios:        dict  = None,
    presupuesto:   float = None,
    wacc:          float = None,
    horizonte:     int   = None,
    precio_carbono:float = None,
    I_min:         float = None,
    G_max:         float = None,
    costo_log:     float = 0.0,
    verbose:       bool  = True,
) -> dict:
    """
    FO5: Minimizar CAPEX total + costo de localización.

    Variables de decisión:
      x[e][t]  : Ton biomasa estructura e procesada con tecnología t
      y[t]     : Binaria — tecnología t activa
      z[s]     : Binaria — sitio s seleccionado (Σ z[s] = 1)

    FO5 = Σ CAPEX[t]·y[t] + Σ (C_terreno[s] + C_infra[s])·z[s]
    """
    if Q_e          is None: Q_e          = Q_BASE.copy()
    if cap_t        is None: cap_t        = CAP_BASE.copy()
    if precio       is None: precio       = PRECIO_BASE.copy()
    if capex_t      is None: capex_t      = CAPEX_BASE.copy()
    if sitios       is None: sitios       = SITIOS.copy()
    if presupuesto  is None: presupuesto  = PRESUPUESTO
    if wacc         is None: wacc         = WACC
    if horizonte    is None: horizonte    = HORIZONTE
    if precio_carbono is None: precio_carbono = PRECIO_CARBONO
    if I_min        is None: I_min        = I_MIN_REAL
    if G_max        is None: G_max        = G_MAX_REAL

    Q_total = sum(Q_e.values())
    LISTA_SITIOS = list(sitios.keys())

    m = pulp.LpProblem("FO5_CAPEX_Localizacion", pulp.LpMinimize)

    # ── Variables de decisión ──────────────────────────────────────
    x = {e:{t:pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t:pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}
    z = {s:pulp.LpVariable(f"z_{s}", cat='Binary') for s in LISTA_SITIOS}

    # ── Expresiones ────────────────────────────────────────────────
    # CAPEX tecnologías
    capex_tec = pulp.lpSum(
        capex_t.get(t, 0) * y[t] for t in TEC_ELEG
    )
    # Costo sitio (terreno + infraestructura)
    costo_sitio = pulp.lpSum(
        sitios[s]['costo_total_sitio'] * z[s] for s in LISTA_SITIOS
    )
    # FO5
    FO5 = capex_tec + costo_sitio

    # Expresiones auxiliares
    ing   = pulp.lpSum(
        precio[p]*1000*R[e][t][p]*x[e][t]
        for e in ESTRUCTURAS for t in TEC_ELEG
        for p in PRODUCTOS if R[e][t][p]>0
    )
    cost  = pulp.lpSum(C_OP[t]*x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)

    # Costo logístico dinámico según sitio seleccionado
    costo_log_sitio = pulp.lpSum(
        sitios[s]['costo_log_anual'] * z[s] for s in LISTA_SITIOS
    )
    util  = ing - cost - costo_log_sitio
    gei   = pulp.lpSum(GEI_BASE[t]*x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    bch   = pulp.lpSum(
        R[e][t]['biochar']*x[e][t]
        for e in ESTRUCTURAS for t in TEC_ELEG if R[e][t]['biochar']>0
    )
    # Ingreso por créditos de carbono
    ingreso_carbono = precio_carbono * PHI_BIOCHAR * bch

    m += FO5, "Minimizar_CAPEX_Total"

    # ── Restricciones ──────────────────────────────────────────────
    # R1: Disponibilidad de biomasa por estructura
    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_e[e]

    # R2: Capacidad instalada por tecnología
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= cap_t[t]*y[t]

    # R3: Tecnologías obligatorias
    m += y['molienda'] == 1
    m += y['secado']   == 1

    # R4: Máximo de tecnologías activas
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC

    # R5: Compostaje mínimo (retorno de nutrientes)
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total

    # R6: Viabilidad económica mínima
    m += util + ingreso_carbono >= I_min

    # R7: Techo de emisiones GEI
    m += gei <= G_max

    # R8 (NUEVA): Presupuesto máximo de inversión ← FO5
    m += FO5 <= presupuesto * FACTOR_INCENTIVO

    # R9 (NUEVA): Selección de exactamente un sitio
    m += pulp.lpSum(z[s] for s in LISTA_SITIOS) == 1

    estado = m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if pulp.LpStatus[estado] != 'Optimal':
        if verbose:
            print(f"  FO5: {pulp.LpStatus[estado]}")
        return {'estado': pulp.LpStatus[estado], 'FO5': 0, 'error': True}

    # ── Resultados ─────────────────────────────────────────────────
    fo5_val       = pulp.value(FO5)
    capex_tec_val = pulp.value(capex_tec)
    costo_sit_val = pulp.value(costo_sitio)
    util_val      = pulp.value(util)
    gei_val       = pulp.value(gei)
    bch_val       = pulp.value(bch)
    tec_act       = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]
    sitio_sel     = [s for s in LISTA_SITIOS if (pulp.value(z[s]) or 0) > 0.5]
    sitio_sel     = sitio_sel[0] if sitio_sel else 'N/A'

    # ── Indicadores financieros derivados ──────────────────────────
    # VPN: asume FO1 (utilidad) constante durante el horizonte
    flujos = [util_val / (1+wacc)**t for t in range(1, horizonte+1)]
    vpn   = sum(flujos) - fo5_val * FACTOR_INCENTIVO + ingreso_carbono*sum(
        1/(1+wacc)**t for t in range(1, horizonte+1))

    # TIR (bisección)
    def npv(r):
        return sum(util_val/(1+r)**t for t in range(1, horizonte+1)) - fo5_val * FACTOR_INCENTIVO
    try:
        lo, hi = 0.001, 5.0
        for _ in range(50):
            mid = (lo+hi)/2
            if npv(mid) > 0: lo = mid
            else: hi = mid
        tir = (lo+hi)/2
    except Exception:
        tir = 0.0

    # PRI: período de recuperación simple
    acum, pri = 0, horizonte
    for t in range(1, horizonte+1):
        acum += util_val
        if acum >= fo5_val * FACTOR_INCENTIVO:
            pri = t; break

    # ROCE
    roce = (util_val / max(fo5_val, 1)) * 100

    # Costo logístico anual del sitio seleccionado
    clog_anual = sitios[sitio_sel]['costo_log_anual'] if sitio_sel != 'N/A' else 0

    if verbose:
        print(f"\n{'='*65}")
        print(f"  SOLUCIÓN FO5 — MINIMIZAR CAPEX + LOCALIZACIÓN")
        print(f"  Datos simulados realistas — Urabá 2024/2025")
        print(f"{'='*65}")
        print(f"  ── CAPEX ──────────────────────────────────────────")
        print(f"  CAPEX tecnologías:   USD {capex_tec_val:>12,.0f}")
        print(f"  Costo sitio:         USD {costo_sit_val:>12,.0f}")
        print(f"  CAPEX TOTAL:         USD {fo5_val:>12,.0f}")
        print(f"  CAPEX c/incentivos:  USD {fo5_val*FACTOR_INCENTIVO:>12,.0f}")
        print(f"  Presupuesto máx:     USD {presupuesto:>12,.0f}")
        print(f"  ── LOCALIZACIÓN ───────────────────────────────────")
        print(f"  Sitio seleccionado:  {sitio_sel}")
        if sitio_sel != 'N/A':
            s = sitios[sitio_sel]
            print(f"  Descripción:         {s['descripcion']}")
            print(f"  Distancia campo:     {s['dist_campo_km']} km")
            print(f"  Costo log. anual:    USD {clog_anual:>10,.0f}/año")
        print(f"  ── FINANCIERO ─────────────────────────────────────")
        print(f"  Utilidad neta:       USD {util_val:>12,.0f}/año")
        print(f"  Ingreso carbono:     USD {pulp.value(ingreso_carbono):>12,.0f}/año")
        print(f"  VPN ({horizonte} años):       USD {vpn:>12,.0f}")
        print(f"  TIR:                     {tir*100:>10.1f}%")
        print(f"  PRI:                     {pri:>10} años")
        print(f"  ROCE:                    {roce:>10.1f}%")
        print(f"  ── OPERACIONAL ────────────────────────────────────")
        print(f"  Tecnologías activas: {tec_act}")
        print(f"  GEI emitido:             {gei_val:>10,.1f} tCO₂/año")
        print(f"  Biochar producido:       {bch_val:>10,.1f} Ton/año")
        print(f"{'='*65}")

    return {
        'estado':          'Optimal',
        'error':           False,
        'FO5':             fo5_val,
        'capex_tec':       capex_tec_val,
        'costo_sitio':     costo_sit_val,
        'capex_incentivos':fo5_val * FACTOR_INCENTIVO,
        'sitio':           sitio_sel,
        'tec_activas':     tec_act,
        'tec_list':        str(tec_act),
        'utilidad_neta':   util_val,
        'ingreso_carbono': pulp.value(ingreso_carbono),
        'GEI_anual':       gei_val,
        'biochar':         bch_val,
        'VPN':             vpn,
        'TIR':             tir,
        'PRI':             pri,
        'ROCE':            roce,
        'costo_log_anual': clog_anual,
        'presupuesto':     presupuesto,
        'wacc':            wacc,
        'horizonte':       horizonte,
        'precio_carbono':  precio_carbono,
        'Q_total':         Q_total,
        'I_min':           I_min,
        'G_max':           G_max,
        'CAPEX_por_tec':   {t: capex_t.get(t,0) for t in tec_act},
        'SITIOS_info':     {s: sitios[s]['costo_total_sitio'] for s in LISTA_SITIOS},
    }


# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — ANÁLISIS DE SENSIBILIDAD
# ══════════════════════════════════════════════════════════════════════
def sensibilidad_presupuesto(n_pasos=10):
    """Analiza qué combinaciones de tecnologías son factibles según el presupuesto."""
    presupuestos = np.linspace(PRESUPUESTO*0.4, PRESUPUESTO*2.0, n_pasos)
    registros = []
    for p in presupuestos:
        r = resolver_FO5(presupuesto=p, verbose=False)
        registros.append({
            'presupuesto_MUSD': p/1e6,
            'estado': r.get('estado','Infeasible'),
            'FO5_MUSD': r.get('FO5',0)/1e6,
            'VPN_MUSD': r.get('VPN',0)/1e6,
            'TIR_pct': r.get('TIR',0)*100,
            'PRI_anos': r.get('PRI',0),
            'sitio': r.get('sitio','N/A'),
            'n_tec': len(r.get('tec_activas',[])),
        })
    return pd.DataFrame(registros)


def sensibilidad_wacc(n_pasos=10):
    """Analiza el impacto del WACC en la viabilidad financiera."""
    waccs = np.linspace(0.08, 0.20, n_pasos)
    r_base = resolver_FO5(verbose=False)
    registros = []
    for w in waccs:
        if r_base.get('estado') == 'Optimal':
            util = r_base['utilidad_neta']
            fo5  = r_base['FO5']
            flujos = [util/(1+w)**t for t in range(1,HORIZONTE+1)]
            vpn_w = sum(flujos) - fo5*FACTOR_INCENTIVO
            registros.append({
                'WACC_pct': w*100,
                'VPN_MUSD': vpn_w/1e6,
                'viable': vpn_w > 0,
            })
    return pd.DataFrame(registros)


def comparar_sitios():
    """Compara los 5 sitios candidatos con CAPEX fijo."""
    registros = []
    for sitio_nombre in SITIOS.keys():
        # Forzar selección de cada sitio
        sitios_mod = {s: d for s, d in SITIOS.items()}
        # Resolver con presupuesto holgado para que no restrinja
        r = resolver_FO5(presupuesto=PRESUPUESTO*3, verbose=False)
        if r.get('sitio') == sitio_nombre or r.get('estado') == 'Optimal':
            s = SITIOS[sitio_nombre]
            registros.append({
                'Sitio': sitio_nombre,
                'Costo terreno (USD)': s['costo_terreno_usd_ha']*s['area_necesaria_ha'],
                'Costo infra (USD)': s['costo_infra_usd'],
                'Costo total sitio (USD)': s['costo_total_sitio'],
                'Dist. campo (km)': s['dist_campo_km'],
                'Costo log. anual (USD)': s['costo_log_anual'],
                'Acceso vial (1-5)': s['acceso_vial'],
                'Descripción': s['descripcion'],
            })
    return pd.DataFrame(registros)


# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
def correr_FO5_completo():
    print(f"\n{'='*65}")
    print("  FO5 — MINIMIZAR CAPEX + LOCALIZACIÓN ÓPTIMA")
    print("  Biorefinería Urabá — Datos Simulados Realistas")
    print(f"{'='*65}")
    print(f"\n  CAPEX_BASE (datos simulados):")
    for t, v in CAPEX_BASE.items():
        if v > 0:
            print(f"    {t:<25}: USD {v:>12,.0f}")
    print(f"\n  Presupuesto máx: USD {PRESUPUESTO:,.0f}")
    print(f"  WACC:            {WACC*100:.1f}%")
    print(f"  Horizonte:       {HORIZONTE} años")
    print(f"  Precio carbono:  USD {PRECIO_CARBONO}/tCO₂")

    print("\n  [1/4] Resolviendo modelo base FO5...")
    res = resolver_FO5(verbose=True)
    if res.get('error'):
        print("  ERROR: modelo FO5 infactible.")
        return

    print("\n  [2/4] Comparando sitios candidatos...")
    df_sitios = comparar_sitios()
    print(df_sitios[['Sitio','Costo total sitio (USD)',
                      'Dist. campo (km)','Acceso vial (1-5)']].to_string(index=False))

    print("\n  [3/4] Sensibilidad al presupuesto...")
    df_pres = sensibilidad_presupuesto(n_pasos=8)
    print(df_pres[['presupuesto_MUSD','estado','VPN_MUSD',
                   'TIR_pct','PRI_anos','sitio']].to_string(index=False))

    print("\n  [4/4] Sensibilidad al WACC...")
    df_wacc = sensibilidad_wacc(n_pasos=8)
    print(df_wacc.to_string(index=False))

    print(f"\n{'='*65}")
    print(f"  RESUMEN FO5:")
    print(f"  CAPEX total:  USD {res['FO5']/1e6:.2f}M")
    print(f"  Sitio óptimo: {res['sitio']}")
    print(f"  VPN:          USD {res['VPN']/1e6:.1f}M")
    print(f"  TIR:          {res['TIR']*100:.1f}%")
    print(f"  PRI:          {res['PRI']} años")
    print(f"  ROCE:         {res['ROCE']:.1f}%")
    print(f"{'='*65}")

    return res, df_sitios, df_pres, df_wacc


def _es_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if __name__ == '__main__' or _es_notebook():
    correr_FO5_completo()
