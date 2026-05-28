"""
milp_core.py — MILP liviano para Streamlit Cloud
Biorefinería Integral Urabá — Grupo ALIADO — UdeA 2025
Versión sin sensibilidades, optimizada para respuesta < 30s
"""
import warnings; warnings.filterwarnings('ignore')
import pulp
import numpy as np

# ── Datos base embebidos (sin depender de datos_base.py) ─────────────
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
TRL = {
    'molienda':9,'secado':9,'compostaje':9,'fermentacion':9,
    'transesterificacion':8,'extraccion_solventes':7,
    'hidrolisis_enzimatica':8,'pirolisis':6,'carbonizacion':6,'gasificacion':5,
}
TEC_ELEG = [t for t in TECNOLOGIAS if TRL[t] >= 4]
MAX_TEC   = 6
PHI_BIOCHAR = 1.65
MU_EMPLEO   = 2.5

# Rendimientos r[e][t][p]
def _build_R():
    r = {e:{t:{p:0.0 for p in PRODUCTOS} for t in TECNOLOGIAS} for e in ESTRUCTURAS}
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
    r['hojas']['extraccion_solventes']['extractos_bioactivos']         = 0.045
    r['hojas']['extraccion_solventes']['pigmentos_antocianinas']       = 0.012
    r['hojas']['fermentacion']['bioetanol']                            = 0.040
    r['hojas']['fermentacion']['biogas_ch4']                           = 0.060
    r['hojas']['compostaje']['compost_biofertilizante']                = 0.400
    r['hojas']['pirolisis']['biochar']                                 = 0.220
    r['hojas']['molienda']['forraje_animal']                           = 0.250
    r['cormo']['extraccion_solventes']['extractos_bioactivos']         = 0.060
    r['cormo']['extraccion_solventes']['almidon_modificado']           = 0.200
    r['cormo']['hidrolisis_enzimatica']['bioplasticos']                = 0.120
    r['cormo']['hidrolisis_enzimatica']['nanocelulosa']                = 0.100
    r['cormo']['fermentacion']['bioetanol']                            = 0.070
    r['cormo']['compostaje']['compost_biofertilizante']                = 0.380
    r['raquis']['molienda']['fibras_tecnicas']                         = 0.200
    r['raquis']['molienda']['fibras_compuestas']                       = 0.150
    r['raquis']['molienda']['papel_kraft']                             = 0.120
    r['raquis']['hidrolisis_enzimatica']['nanocelulosa']               = 0.180
    r['raquis']['hidrolisis_enzimatica']['bioadhesivos']               = 0.080
    r['raquis']['pirolisis']['biochar']                                = 0.300
    r['raquis']['carbonizacion']['biochar']                            = 0.350
    r['raquis']['compostaje']['compost_biofertilizante']               = 0.300
    return r

R = _build_R()

# ── Parámetros base SD (escala real) ─────────────────────────────────
SD_DEFAULT = {
    'Q_gen_anual':    1_265_116.0,
    'Q_total_anual':    929_692.0,
    'eta_cadena':           0.420,
    'GEI_base':          7_796.6,
    'fertilidad':            0.449,
    'superficie':         36_932.0,
}

def get_params(sd=None, eta=None, precio_factor=1.0, gei_factor=1.0):
    """Construye Q_BASE, CAP_BASE, PRECIO_BASE, GEI_BASE, C_OP desde parámetros."""
    if sd is None: sd = SD_DEFAULT
    eta_c   = eta if eta is not None else sd['eta_cadena']
    Q_total = sd['Q_total_anual']

    ALPHA = {'pseudotallo':0.8006,'hojas':0.1156,'cormo':0.0578,'raquis':0.0260}
    Q_BASE = {e: ALPHA[e] * Q_total for e in ESTRUCTURAS}

    # Capacidades proporcionales a biomasa recolectada
    f = Q_total / 929_692.0   # factor vs base
    CAP_BASE = {
        'molienda':              min(Q_total * 0.50, 500_000),
        'secado':                min(Q_total * 0.40, 400_000),
        'compostaje':            min(Q_total * 0.35, 350_000),
        'fermentacion':          min(Q_total * 0.15, 150_000),
        'transesterificacion':   min(Q_total * 0.05,  50_000),
        'extraccion_solventes':  min(Q_total * 0.08,  80_000),
        'hidrolisis_enzimatica': min(Q_total * 0.10, 100_000),
        'pirolisis':             min(Q_total * 0.08,  80_000),
        'carbonizacion':         min(Q_total * 0.08,  80_000),
        'gasificacion':          0.0,
    }
    PRECIO_BASE = {p: v * precio_factor for p, v in {
        'extractos_bioactivos':45.00,'pigmentos_antocianinas':30.00,
        'nanocelulosa':7.10,'bioplasticos':3.50,'biopeliculas':3.00,
        'fibras_compuestas':2.50,'bioadhesivos':4.00,'fibras_tecnicas':0.80,
        'bioetanol':0.65,'biogas_ch4':0.05,'biochar':0.35,
        'compost_biofertilizante':0.15,'almidon_modificado':0.45,
        'papel_kraft':0.55,'forraje_animal':0.30,'biocombustible':2.11,
    }.items()}
    GEI_OP = {t: v * gei_factor for t, v in {
        'molienda':0.015,'secado':0.045,'compostaje':0.080,'fermentacion':0.120,
        'transesterificacion':0.095,'extraccion_solventes':0.180,
        'hidrolisis_enzimatica':0.140,'pirolisis':0.280,'carbonizacion':0.310,
        'gasificacion':0.001,
    }.items()}
    C_OP = {
        'molienda':12.5,'secado':30.0,'compostaje':65.0,'fermentacion':200.0,
        'transesterificacion':100.0,'extraccion_solventes':330.0,
        'hidrolisis_enzimatica':400.0,'pirolisis':197.5,'carbonizacion':197.5,
        'gasificacion':9999.0,
    }
    EMP_BASE = {
        'molienda':0.0080,'secado':0.0060,'compostaje':0.0280,'fermentacion':0.0150,
        'transesterificacion':0.0120,'extraccion_solventes':0.0200,
        'hidrolisis_enzimatica':0.0180,'pirolisis':0.0100,'carbonizacion':0.0100,
        'gasificacion':0.0001,
    }
    # G_max dinámico
    gei_min = (CAP_BASE['molienda']*GEI_OP['molienda'] +
               CAP_BASE['secado']*GEI_OP['secado'] +
               Q_total*0.10*GEI_OP['compostaje'])
    G_MAX = max(gei_min * 5, 150_000)
    I_MIN = max(Q_total * 50, 50_000_000)   # ~USD 50/Ton mínimo

    return Q_BASE, CAP_BASE, PRECIO_BASE, GEI_OP, C_OP, EMP_BASE, G_MAX, I_MIN


def resolver_milp(
    objetivo:      str   = 'FO1',   # 'FO1','FO2','FO3','FO4','compromiso'
    sd_params:     dict  = None,
    eta:           float = None,
    precio_factor: float = 1.0,
    gei_factor:    float = 1.0,
    phi:           float = 1.65,
    mu:            float = 2.5,
    w:             list  = None,    # pesos [w1,w2,w3,w4] para compromiso
) -> dict:
    """
    Resuelve el MILP con los parámetros dados.
    Retorna dict con todos los resultados para el dashboard.
    """
    Q_BASE, CAP_BASE, PRECIO_BASE, GEI_OP, C_OP, EMP_BASE, G_MAX, I_MIN = \
        get_params(sd_params, eta, precio_factor, gei_factor)

    Q_total = sum(Q_BASE.values())
    Q_gen   = Q_total / (eta if eta else SD_DEFAULT['eta_cadena'])

    # ── Modelo
    sentido = pulp.LpMinimize if objetivo == 'FO2' else pulp.LpMaximize
    if objetivo == 'compromiso': sentido = pulp.LpMinimize
    m = pulp.LpProblem(f"MILP_{objetivo}", sentido)

    x = {e:{t:pulp.LpVariable(f"x_{e}_{t}", lowBound=0)
             for t in TEC_ELEG} for e in ESTRUCTURAS}
    y = {t:pulp.LpVariable(f"y_{t}", cat='Binary') for t in TEC_ELEG}

    # Expresiones
    ing  = pulp.lpSum(PRECIO_BASE[p]*1000*R[e][t][p]*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      for p in PRODUCTOS if R[e][t][p]>0)
    cost = pulp.lpSum(C_OP[t]*x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    gei  = pulp.lpSum(GEI_OP[t]*x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    emp  = pulp.lpSum(EMP_BASE[t]*x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    bio  = pulp.lpSum(x[e][t] for e in ESTRUCTURAS for t in TEC_ELEG)
    bch  = pulp.lpSum(R[e][t]['biochar']*x[e][t]
                      for e in ESTRUCTURAS for t in TEC_ELEG
                      if R[e][t]['biochar']>0)
    util = ing - cost

    # Función objetivo
    if objetivo == 'FO1':
        m += util
    elif objetivo == 'FO2':
        m += gei - phi * bch
    elif objetivo == 'FO3':
        m += emp + 0.003 * bio
    elif objetivo == 'FO4':
        m += bio
    elif objetivo == 'compromiso':
        if w is None: w = [0.25, 0.25, 0.25, 0.25]
        # Rangos aproximados para normalización
        U_MAX, U_MIN = 356_532_688, 50_000_000
        G_MAX_OBJ, G_MIN_OBJ = 73_631, -18_861
        E_MAX, E_MIN = 19_532, 4_607
        B_MAX, B_MIN = Q_total, Q_total*0.27
        rng = np.array([U_MAX-U_MIN, G_MAX_OBJ-G_MIN_OBJ, E_MAX-E_MIN, B_MAX-B_MIN]) + 1e-9
        m += (w[0]*(U_MAX - util)/rng[0] +
              w[1]*(gei - phi*bch - G_MIN_OBJ)/rng[1] +
              w[2]*(E_MAX - emp)/rng[2] +
              w[3]*(B_MAX - bio)/rng[3])

    # Restricciones
    for e in ESTRUCTURAS:
        m += pulp.lpSum(x[e][t] for t in TEC_ELEG) <= Q_BASE[e]
    for t in TEC_ELEG:
        m += pulp.lpSum(x[e][t] for e in ESTRUCTURAS) <= CAP_BASE[t]*y[t]
    m += y['molienda'] == 1
    m += y['secado']   == 1
    m += pulp.lpSum(y[t] for t in TEC_ELEG) <= MAX_TEC
    m += pulp.lpSum(x[e]['compostaje'] for e in ESTRUCTURAS) >= 0.10*Q_total
    m += util >= I_MIN
    m += gei  <= G_MAX

    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=25))

    if pulp.LpStatus[m.status] != 'Optimal':
        return {'estado': pulp.LpStatus[m.status], 'error': True}

    fo1_v = pulp.value(util)
    fo2_v = pulp.value(gei) - phi * pulp.value(bch)
    fo3_v = pulp.value(emp) + 0.003 * pulp.value(bio)
    fo4_v = min(pulp.value(bio)/Q_total*100, 100)
    tec   = [t for t in TEC_ELEG if (pulp.value(y[t]) or 0) > 0.5]

    # Empleo completo
    emp_br  = pulp.value(emp)
    emp_cam = 0.003 * pulp.value(bio)
    emp_aco = 0.002 * Q_total + 4 * 2.0
    emp_dir = emp_br + emp_cam + emp_aco
    emp_tot = emp_dir * (1 + mu)

    # Producción por producto
    prod = {p: sum(R[e][t][p]*(pulp.value(x[e][t]) or 0)
                   for e in ESTRUCTURAS for t in TEC_ELEG)
            for p in PRODUCTOS}

    alpha_real = min(pulp.value(bio)/Q_gen, 1.0) if Q_gen > 0 else 0
    alpha_BR   = min(pulp.value(bio)/Q_total, 1.0) if Q_total > 0 else 0

    return {
        'estado':        'Optimal',
        'error':         False,
        'FO1':           fo1_v,
        'FO2':           fo2_v,
        'FO3':           fo3_v,
        'FO4':           fo4_v,
        'ingreso_bruto': pulp.value(ing),
        'costo_op':      pulp.value(cost),
        'GEI_bruto':     pulp.value(gei),
        'GEI_secuestro': phi * pulp.value(bch),
        'biochar':       pulp.value(bch),
        'emp_biorref':   emp_br,
        'emp_campo':     emp_cam,
        'emp_acopio':    emp_aco,
        'emp_directo':   emp_dir,
        'emp_total':     emp_tot,
        'biomasa_proc':  pulp.value(bio),
        'alpha_real':    alpha_real,
        'alpha_BR':      alpha_BR,
        'Q_total':       Q_total,
        'Q_gen':         Q_gen,
        'tec_activas':   tec,
        'produccion':    prod,
        'I_MIN':         I_MIN,
        'G_MAX':         G_MAX,
    }
