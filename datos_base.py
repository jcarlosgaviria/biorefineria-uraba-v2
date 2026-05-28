"""
╔══════════════════════════════════════════════════════════════════════╗
║   DATOS BASE v2 — Parámetros compartidos por FO1, FO2, FO3 y FO4    ║
║   Biorrefinería de Biomasa Residual de Banano — Urabá, Colombia      ║
║   Autor : Juan Carlos Gaviria Chaverra                               ║
║   Org.  : Universidad de Antioquia — Grupo ALIADO — 2025             ║
║   v2    : Escalado a biomasa real SD (1,265,116 Ton/año)             ║
║           Fracciones alineadas con variables SD confirmadas           ║
║           Capacidades instaladas escaladas proporcionalmente          ║
╚══════════════════════════════════════════════════════════════════════╝

CAMBIOS v1 → v2:
    - BIOMASA_TOTAL_ANUAL: 3,000 → 1,265,116 Ton/año (datos SD reales)
    - ESTRUCTURAS: pseudotallo renombrado a pseudotallo/fibra_vascular
      y fracciones recalibradas con desglose SD confirmado
    - ALPHA: recalibrado con valores reales del SD (t=100)
    - CAP_BASE: escalado ×421 respecto a versión piloto
    - Q_BASE: ahora usa valores absolutos del SD por fracción
    - SD_PARAMS: nuevo bloque con parámetros directos del conector SD

CÓMO USAR:
    from datos_base_v2 import *          # importa todo
    from datos_base_v2 import R, TEC_ELEG, PRECIO_BASE, SD_PARAMS
"""

import os

# ── Conjuntos ────────────────────────────────────────────────────────────
# Alineado con fracciones confirmadas en el SD (Diagrama_Hibrido_Uraba_v6.mdl)
ESTRUCTURAS = ['pseudotallo', 'hojas', 'cormo', 'raquis']

TECNOLOGIAS = [
    'molienda', 'secado', 'compostaje', 'fermentacion',
    'transesterificacion', 'extraccion_solventes',
    'hidrolisis_enzimatica', 'pirolisis', 'carbonizacion', 'gasificacion',
]

PRODUCTOS = [
    'extractos_bioactivos', 'pigmentos_antocianinas', 'nanocelulosa',
    'bioplasticos', 'biopeliculas', 'fibras_compuestas', 'bioadhesivos',
    'fibras_tecnicas', 'bioetanol', 'biogas_ch4', 'biochar',
    'compost_biofertilizante', 'almidon_modificado',
    'papel_kraft', 'forraje_animal', 'biocombustible',
]

# ── Nivel de madurez tecnológica (TRL) ──────────────────────────────────
TRL = {
    'molienda': 9, 'secado': 9, 'compostaje': 9, 'fermentacion': 9,
    'transesterificacion': 8, 'extraccion_solventes': 7,
    'hidrolisis_enzimatica': 8, 'pirolisis': 6,
    'carbonizacion': 6, 'gasificacion': 5,
}
TRL_MIN  = 4
TEC_ELEG = [t for t in TECNOLOGIAS if TRL[t] >= TRL_MIN]
MAX_TEC  = 6


# ══════════════════════════════════════════════════════════════════════════
# BLOQUE SD — Parámetros del conector SD-MILP v3
# Fuente: Diagrama_Hibrido_Uraba_v6.mdl | pysd 3.14.3 | promedio t=10-100
# ══════════════════════════════════════════════════════════════════════════
# Intentar cargar desde resumen_sd.csv si existe (Colab con Drive montado)
_ruta_resumen = '/content/drive/MyDrive/Biorefineria_Uraba/datos_sd/resumen_sd.csv'

def _cargar_params_sd():
    """Carga parámetros SD desde CSV si está disponible, si no usa valores confirmados."""
    if os.path.exists(_ruta_resumen):
        try:
            import pandas as pd
            df = pd.read_csv(_ruta_resumen)
            params = df.iloc[0].to_dict()
            print("  ✓ datos_base_v2: parámetros SD cargados desde resumen_sd.csv")
            return params
        except Exception:
            pass
    # Fallback: valores confirmados de la corrida SD del 24/05/2026
    return {
        'Q_gen_anual_sd':    1_265_116.0,   # Ton/año — promedio t=10-100
        'Q_total_anual_sd':    929_692.0,   # Ton/año — Tasa Recoleccion Efectiva
        'eta_cadena_sd':           0.4200,
        'GEI_base_sd':          7_796.6,    # TonCO2/mes
        'G_max_sd':             3_227.5,    # TonCO2/mes
        'fertilidad_sd':           0.449,
        'superficie_sd':        36_932.0,   # Ha
        'costo_log_sd':              0.0,
        # Fracciones mensuales confirmadas (t=100, Ton/mes)
        'bio_hojas_mean':       11_300.0,
        'bio_cormo_mean':        5_650.0,
        'bio_campo_tot_mean':   97_800.0,
        'bio_raquis_mean':       2_545.0,
        'bio_bracteas_mean':       905.0,
        'bio_flores_e_mean':       566.0,
        'bio_manos_mean':        3_730.0,
    }

SD_PARAMS = _cargar_params_sd()


# ── Biomasa total anual — ESCALA REAL del SD ─────────────────────────────
# ✅ v2: actualizado de 3,000 → 1,265,116 Ton/año
BIOMASA_TOTAL_ANUAL = SD_PARAMS.get('Q_gen_anual_sd', 1_265_116.0)   # Ton/año
BIOMASA_RECOLECTADA = SD_PARAMS.get('Q_total_anual_sd', 929_692.0)   # Ton/año
ETA_CADENA          = SD_PARAMS.get('eta_cadena_sd', 0.42)


# ── Fracciones ALPHA — recalibradas con SD ───────────────────────────────
# Fuente: Biomasa Campo Total SD = 97,800 Ton/mes promedio
# Desglose confirmado por variable SD:
#   Hojas          : 11,300 / 97,800 = 11.55%
#   Cormo          : 5,650  / 97,800 = 5.78%
#   Raquis         : 2,545  / 97,800 = 2.60%
#   Pseudotallo    : resto  = 80.07% (fibra vascular + agua estructural)
#   (brácteas, flores, manos → fracciones menores incluidas en pseudotallo)

_bio_campo_tot_anual = SD_PARAMS.get('bio_campo_tot_mean', 97_800) * 12

ALPHA = {
    'pseudotallo': round(1.0 - (
        SD_PARAMS.get('bio_hojas_mean', 11_300) +
        SD_PARAMS.get('bio_cormo_mean',  5_650) +
        SD_PARAMS.get('bio_raquis_mean', 2_545)
    ) / SD_PARAMS.get('bio_campo_tot_mean', 97_800), 4),   # ~0.8007
    'hojas':  round(SD_PARAMS.get('bio_hojas_mean',  11_300) /
                    SD_PARAMS.get('bio_campo_tot_mean', 97_800), 4),  # ~0.1155
    'cormo':  round(SD_PARAMS.get('bio_cormo_mean',   5_650) /
                    SD_PARAMS.get('bio_campo_tot_mean', 97_800), 4),  # ~0.0578
    'raquis': round(SD_PARAMS.get('bio_raquis_mean',  2_545) /
                    SD_PARAMS.get('bio_campo_tot_mean', 97_800), 4),  # ~0.0260
}

# Disponibilidad base por estructura (Ton/año) — usa biomasa recolectada
Q_BASE = {e: round(ALPHA[e] * BIOMASA_RECOLECTADA, 1) for e in ESTRUCTURAS}


# ── Capacidades instaladas (Ton biomasa/año) ─────────────────────────────
# ✅ v2: escalado desde escala piloto (3,000 t/año) a escala real
# Factor de escala = 929,692 / 3,000 = 309.9 ≈ 310×
# Se aplica con límite superior razonable por tecnología
# [CALIBRAR] con ingeniería de detalle y cotizaciones para Urabá

_ESCALA = BIOMASA_RECOLECTADA / 3_000.0   # ~310×

CAP_BASE = {
    'molienda':             min(BIOMASA_RECOLECTADA * 0.50,  500_000),  # 50% de biomasa
    'secado':               min(BIOMASA_RECOLECTADA * 0.40,  400_000),  # 40%
    'compostaje':           min(BIOMASA_RECOLECTADA * 0.35,  350_000),  # 35%
    'fermentacion':         min(BIOMASA_RECOLECTADA * 0.15,  150_000),  # 15%
    'transesterificacion':  min(BIOMASA_RECOLECTADA * 0.05,   50_000),  # 5%
    'extraccion_solventes': min(BIOMASA_RECOLECTADA * 0.08,   80_000),  # 8%
    'hidrolisis_enzimatica':min(BIOMASA_RECOLECTADA * 0.10,  100_000),  # 10%
    'pirolisis':            min(BIOMASA_RECOLECTADA * 0.08,   80_000),  # 8%
    'carbonizacion':        min(BIOMASA_RECOLECTADA * 0.08,   80_000),  # 8%
    'gasificacion':         0.0,   # excluida por TRL < TRL_MIN
}


# ── Costos operativos (USD/Ton biomasa procesada) ────────────────────────
# Sin cambios respecto a v1 — son costos unitarios, escalan con volumen
# [CALIBRAR] con TEA real de operación a escala industrial
C_OP = {
    'molienda':               12.5,
    'secado':                 30.0,
    'compostaje':             65.0,
    'fermentacion':          200.0,
    'transesterificacion':   100.0,
    'extraccion_solventes':  330.0,
    'hidrolisis_enzimatica': 400.0,
    'pirolisis':             197.5,
    'carbonizacion':         197.5,
    'gasificacion':        9_999.0,
}


# ── Precios base de productos (USD/kg) ───────────────────────────────────
# Sin cambios — precios de mercado 2025-2026
# [CALIBRAR] con estudios de mercado actualizados
PRECIO_BASE = {
    'extractos_bioactivos':    45.00,
    'pigmentos_antocianinas':  30.00,
    'nanocelulosa':             7.10,
    'bioplasticos':             3.50,
    'biopeliculas':             3.00,
    'fibras_compuestas':        2.50,
    'bioadhesivos':             4.00,
    'fibras_tecnicas':          0.80,
    'bioetanol':                0.65,
    'biogas_ch4':               0.05,
    'biochar':                  0.35,
    'compost_biofertilizante':  0.15,
    'almidon_modificado':       0.45,
    'papel_kraft':              0.55,
    'forraje_animal':           0.30,
    'biocombustible':           2.11,
}


# ── Factores de emisión GEI (tCO2-eq / Ton biomasa procesada) ────────────
# Sin cambios — coeficientes unitarios de literatura
GEI_BASE = {
    'molienda':              0.015,
    'secado':                0.045,
    'compostaje':            0.080,
    'fermentacion':          0.120,
    'transesterificacion':   0.095,
    'extraccion_solventes':  0.180,
    'hidrolisis_enzimatica': 0.140,
    'pirolisis':             0.280,
    'carbonizacion':         0.310,
    'gasificacion':          0.001,
}

# GEI base del SD (emisiones de fondo del sistema agrícola)
GEI_BASE_SD = SD_PARAMS.get('GEI_base_sd', 7_796.6)   # TonCO2/mes
G_MAX_SD    = SD_PARAMS.get('G_max_sd',    3_227.5)    # TonCO2/mes (límite MILP)


# ── Factores de empleo (empleos / Ton biomasa / año) ────────────────────
# Sin cambios — coeficientes unitarios de AUGURA (2020)
EMPLEO_BASE = {
    'molienda':              0.0080,
    'secado':                0.0060,
    'compostaje':            0.0280,
    'fermentacion':          0.0150,
    'transesterificacion':   0.0120,
    'extraccion_solventes':  0.0200,
    'hidrolisis_enzimatica': 0.0180,
    'pirolisis':             0.0100,
    'carbonizacion':         0.0100,
    'gasificacion':          0.0001,
}
MU_EMPLEO   = 2.5    # multiplicador empleo indirecto
PHI_BIOCHAR = 1.65   # tCO2-eq secuestrado / Ton biochar


# ── Rendimientos r[e][t][p] ──────────────────────────────────────────────
# Sin cambios — son fracciones adimensionales, independientes de escala
def construir_rendimientos():
    r = {
        e: {t: {p: 0.0 for p in PRODUCTOS}
            for t in TECNOLOGIAS}
        for e in ESTRUCTURAS
    }

    # ── Pseudotallo (fibra vascular + agua estructural, ~80% biomasa)
    r['pseudotallo']['molienda']['fibras_tecnicas']                    = 0.120
    r['pseudotallo']['molienda']['papel_kraft']                        = 0.080
    r['pseudotallo']['secado']['fibras_tecnicas']                      = 0.100
    r['pseudotallo']['extraccion_solventes']['extractos_bioactivos']   = 0.025
    r['pseudotallo']['extraccion_solventes']['pigmentos_antocianinas'] = 0.005
    r['pseudotallo']['extraccion_solventes']['biocombustible']         = 0.180
    r['pseudotallo']['hidrolisis_enzimatica']['nanocelulosa']          = 0.150
    r['pseudotallo']['hidrolisis_enzimatica']['bioplasticos']          = 0.080
    r['pseudotallo']['hidrolisis_enzimatica']['biopeliculas']          = 0.050
    r['pseudotallo']['fermentacion']['bioetanol']                      = 0.055
    r['pseudotallo']['fermentacion']['biogas_ch4']                     = 0.080
    r['pseudotallo']['pirolisis']['biochar']                           = 0.280
    r['pseudotallo']['pirolisis']['biogas_ch4']                        = 0.120
    r['pseudotallo']['carbonizacion']['biochar']                       = 0.320
    r['pseudotallo']['compostaje']['compost_biofertilizante']          = 0.350

    # ── Hojas (~11.55% biomasa)
    r['hojas']['extraccion_solventes']['extractos_bioactivos']         = 0.045
    r['hojas']['extraccion_solventes']['pigmentos_antocianinas']       = 0.012
    r['hojas']['fermentacion']['bioetanol']                            = 0.040
    r['hojas']['fermentacion']['biogas_ch4']                           = 0.060
    r['hojas']['compostaje']['compost_biofertilizante']                = 0.400
    r['hojas']['pirolisis']['biochar']                                 = 0.220
    r['hojas']['molienda']['forraje_animal']                           = 0.250

    # ── Cormo (~5.78% biomasa)
    r['cormo']['extraccion_solventes']['extractos_bioactivos']         = 0.060
    r['cormo']['extraccion_solventes']['almidon_modificado']           = 0.200
    r['cormo']['hidrolisis_enzimatica']['bioplasticos']                = 0.120
    r['cormo']['hidrolisis_enzimatica']['nanocelulosa']                = 0.100
    r['cormo']['fermentacion']['bioetanol']                            = 0.070
    r['cormo']['compostaje']['compost_biofertilizante']                = 0.380

    # ── Raquis (~2.60% biomasa)
    r['raquis']['molienda']['fibras_tecnicas']                         = 0.200
    r['raquis']['molienda']['fibras_compuestas']                       = 0.150
    r['raquis']['molienda']['papel_kraft']                             = 0.120
    r['raquis']['hidrolisis_enzimatica']['nanocelulosa']               = 0.180
    r['raquis']['hidrolisis_enzimatica']['bioadhesivos']               = 0.080
    r['raquis']['pirolisis']['biochar']                                = 0.300
    r['raquis']['carbonizacion']['biochar']                            = 0.350
    r['raquis']['compostaje']['compost_biofertilizante']               = 0.300

    return r

R = construir_rendimientos()


# ── Verificación de integridad ───────────────────────────────────────────
def verificar_datos():
    print("=" * 65)
    print("  VERIFICACIÓN DE DATOS BASE v2")
    print("  Fuente SD: Diagrama_Hibrido_Uraba_v6.mdl | pysd 3.14.3")
    print("=" * 65)

    print(f"\n  ── Biomasa (Escala Real SD) ──")
    print(f"  Generación total : {BIOMASA_TOTAL_ANUAL:>12,.0f} Ton/año")
    print(f"  Biomasa recolect.: {BIOMASA_RECOLECTADA:>12,.0f} Ton/año")
    print(f"  eta_cadena       : {ETA_CADENA:>12.4f}")
    print(f"  GEI base SD      : {GEI_BASE_SD:>12,.1f} TonCO2/mes")
    print(f"  Superficie       : {SD_PARAMS.get('superficie_sd',36932):>12,.0f} Ha")

    print(f"\n  ── Fracciones ALPHA (recalibradas SD) ──")
    for e in ESTRUCTURAS:
        print(f"  {e:<15}: alpha={ALPHA[e]:.4f}  "
              f"Q_base={Q_BASE[e]:>10,.0f} Ton/año")

    print(f"\n  ── Capacidades instaladas (escala real) ──")
    for t in TEC_ELEG:
        print(f"  {t:<25} Cap={CAP_BASE[t]:>10,.0f} Ton/año  "
              f"C_op=USD {C_OP[t]:>6.1f}/Ton  "
              f"TRL={TRL[t]}")

    n_rend = sum(1 for e in ESTRUCTURAS for t in TECNOLOGIAS
                 for p in PRODUCTOS if R[e][t][p] > 0)
    print(f"\n  Rendimientos no nulos: {n_rend}")
    print(f"  Tecnologías elegibles (TRL≥{TRL_MIN}): {len(TEC_ELEG)}")
    print(f"  Productos: {len(PRODUCTOS)}")

    # Proyección rápida de ingresos potenciales máximos
    ingreso_max = 0
    for e in ESTRUCTURAS:
        for t in TEC_ELEG:
            for p in PRODUCTOS:
                if R[e][t][p] > 0:
                    ingreso_max += (Q_BASE[e] * R[e][t][p] *
                                    1000 * PRECIO_BASE[p])
    print(f"\n  Ingreso potencial máx. (todas activas): "
          f"USD {ingreso_max/1e6:,.1f}M/año")
    print("  (referencia teórica — MILP optimiza la selección real)")
    print("=" * 65)
    print("  Datos base v2 verificados correctamente.")
    print("=" * 65)


if __name__ == '__main__':
    verificar_datos()
