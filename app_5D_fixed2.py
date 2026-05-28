"""
╔══════════════════════════════════════════════════════════════════════╗
║   app_5D.py — Biorefinería Integral Urabá — VERSIÓN 5D              ║
║   Dashboard SD-MILP con FO1-FO5 + Análisis CAPEX + Localización     ║
║   Universidad de Antioquia · Grupo ALIADO · 2025                    ║
║   Estilo: Dark Mode · Verde Selva · Dorado · Alto Contraste         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time, os

# ── Configuración ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biorefinería Urabá · SD-MILP 5D",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Global — idéntico a v1 ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
:root {
  --verde-oscuro:#0A1F0E; --verde-base:#1B4D2E; --verde-med:#2E7D32;
  --verde-vivo:#4CAF50; --verde-cl:#A5D6A7; --dorado:#FFD700;
  --dorado-cl:#FFF176; --dorado-osc:#F57F17; --blanco:#F8FFF8;
  --gris-cl:#B0BEC5; --gris-med:#546E7A; --negro:#060E08;
  --azul-acento:#00BCD4; --rojo-acento:#FF5252; --naran-acento:#FF6D00;
  --purp:#CE93D8; --teal:#00897B;
}
.stApp {
  background:linear-gradient(135deg,var(--negro) 0%,var(--verde-oscuro) 50%,#0D1F10 100%) !important;
  font-family:'DM Sans',sans-serif !important;
}
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#060E08 0%,#0A1F0E 100%) !important;
  border-right:1px solid var(--verde-med) !important;
}
[data-testid="stSidebar"] * { color:var(--blanco) !important; }
.hero-header {
  background:linear-gradient(135deg,var(--verde-base) 0%,var(--negro) 100%);
  border:1px solid var(--verde-med); border-left:5px solid var(--dorado);
  border-radius:12px; padding:2rem 2.5rem; margin-bottom:1.5rem;
  position:relative; overflow:hidden;
}
.hero-header::before {
  content:''; position:absolute; top:-50%; right:-10%;
  width:300px; height:300px;
  background:radial-gradient(circle,rgba(255,215,0,0.08) 0%,transparent 70%);
  border-radius:50%;
}
.hero-title {
  font-family:'Syne',sans-serif !important; font-size:2rem !important;
  font-weight:800 !important; color:var(--blanco) !important;
  letter-spacing:-0.02em; line-height:1.1; margin:0;
}
.hero-subtitle {
  font-family:'Space Mono',monospace !important; font-size:0.75rem !important;
  color:var(--dorado) !important; letter-spacing:0.15em;
  text-transform:uppercase; margin-top:0.5rem;
}
.hero-badge {
  display:inline-block; background:rgba(255,215,0,0.15);
  border:1px solid var(--dorado); color:var(--dorado);
  padding:0.2rem 0.8rem; border-radius:20px;
  font-family:'Space Mono',monospace; font-size:0.65rem;
  letter-spacing:0.1em; margin-right:0.5rem; margin-top:0.8rem;
}
.kpi-card {
  background:linear-gradient(135deg,rgba(27,77,46,0.4) 0%,rgba(6,14,8,0.8) 100%);
  border:1px solid rgba(76,175,80,0.3); border-top:3px solid;
  border-radius:12px; padding:1.4rem 1.2rem; margin-bottom:0.5rem;
}
.kpi-label {
  font-family:'Space Mono',monospace; font-size:0.6rem;
  letter-spacing:0.15em; text-transform:uppercase;
  color:var(--gris-cl); margin-bottom:0.5rem;
}
.kpi-value {
  font-family:'Syne',sans-serif; font-size:1.8rem;
  font-weight:800; line-height:1; margin-bottom:0.3rem;
}
.kpi-delta { font-family:'DM Sans',sans-serif; font-size:0.75rem; color:var(--gris-cl); }
.section-title {
  font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
  color:var(--blanco); letter-spacing:0.05em; text-transform:uppercase;
  border-left:3px solid var(--dorado); padding-left:0.8rem;
  margin:1.5rem 0 1rem 0;
}
.v2-badge {
  display:inline-block; background:rgba(255,109,0,0.2);
  border:2px solid #FF6D00; color:#FF6D00;
  padding:0.3rem 1rem; border-radius:20px;
  font-family:'Space Mono',monospace; font-size:0.75rem;
  font-weight:700; letter-spacing:0.1em; margin-left:1rem;
}
h1,h2,h3,h4,p,span,div { color:var(--blanco); }
.stMarkdown p { color:var(--blanco) !important; }
.stButton > button {
  background:linear-gradient(135deg,var(--verde-med),var(--verde-vivo)) !important;
  color:white !important; border:none !important; border-radius:8px !important;
  font-family:'Syne',sans-serif !important; font-weight:700 !important;
  letter-spacing:0.05em !important; padding:0.6rem 1.5rem !important;
  transition:all 0.3s !important;
}
.stButton > button:hover {
  background:linear-gradient(135deg,var(--dorado-osc),var(--dorado)) !important;
  color:var(--negro) !important; transform:translateY(-2px) !important;
  box-shadow:0 8px 25px rgba(255,215,0,0.3) !important;
}
.stSelectbox > div > div {
  background:rgba(27,77,46,0.4) !important;
  border:1px solid var(--verde-med) !important;
  color:var(--blanco) !important; border-radius:8px !important;
}
.stRadio > div { gap:0.5rem !important; }
.stRadio label {
  background:rgba(27,77,46,0.3) !important;
  border:1px solid rgba(76,175,80,0.3) !important;
  border-radius:8px !important; padding:0.4rem 0.8rem !important;
  color:var(--blanco) !important; transition:all 0.2s !important;
}
.stRadio label:hover {
  border-color:var(--dorado) !important;
  background:rgba(255,215,0,0.1) !important;
}
[data-testid="stMetric"] {
  background:rgba(27,77,46,0.3) !important;
  border:1px solid rgba(76,175,80,0.2) !important;
  border-radius:10px !important; padding:1rem !important;
}
[data-testid="stMetricLabel"] { color:var(--gris-cl) !important; }
[data-testid="stMetricValue"] {
  color:var(--dorado) !important;
  font-family:'Syne',sans-serif !important; font-weight:800 !important;
}
[data-testid="stMetricDelta"] { color:var(--verde-cl) !important; }
.stTabs [data-baseweb="tab-list"] {
  background:rgba(6,14,8,0.8) !important;
  border-radius:10px !important; padding:0.3rem !important; gap:0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--gris-cl) !important;
  border-radius:8px !important; font-family:'Space Mono',monospace !important;
  font-size:0.7rem !important; letter-spacing:0.08em !important;
}
.stTabs [aria-selected="true"] {
  background:var(--verde-med) !important; color:white !important;
}
hr { border-color:rgba(76,175,80,0.2) !important; }
.sidebar-logo { text-align:center; padding:1rem 0 0.5rem 0;
  border-bottom:1px solid rgba(76,175,80,0.3); margin-bottom:1rem; }
.sidebar-logo-text {
  font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800;
  color:var(--dorado) !important; letter-spacing:0.05em;
}
.sidebar-sub {
  font-family:'Space Mono',monospace; font-size:0.6rem;
  color:var(--verde-cl) !important; letter-spacing:0.1em; text-transform:uppercase;
}
.capex-card {
  background:linear-gradient(135deg,rgba(230,81,0,0.2) 0%,rgba(6,14,8,0.8) 100%);
  border:1px solid rgba(255,109,0,0.4); border-top:3px solid #FF6D00;
  border-radius:12px; padding:1.2rem; margin-bottom:0.5rem;
}
.sitio-card {
  background:rgba(10,31,14,0.8); border:1px solid rgba(76,175,80,0.3);
  border-radius:10px; padding:1rem; margin-bottom:0.4rem;
  transition:all 0.3s;
}
.sitio-optimo {
  border:2px solid #FFD700 !important;
  background:rgba(255,215,0,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Layout base para gráficos ─────────────────────────────────────────
def layout_base(height=400, title_text='', title_x=0.5, title_size=15,
                xaxis_title='', yaxis_title='', hovermode='closest',
                showlegend=False):
    layout = dict(
        paper_bgcolor='rgba(6,14,8,0.0)',
        plot_bgcolor='rgba(10,31,14,0.4)',
        font=dict(family='DM Sans', color='#F8FFF8', size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode=hovermode,
        showlegend=showlegend,
        colorway=['#4CAF50','#00BCD4','#FFD700','#FF6D00','#FF5252','#CE93D8'],
        legend=dict(bgcolor='rgba(6,14,8,0.7)', bordercolor='rgba(76,175,80,0.3)',
                    borderwidth=1, font=dict(color='#F8FFF8')),
    )
    if title_text:
        layout['title'] = dict(text=title_text, x=title_x,
                                font=dict(family='Syne', size=title_size, color='#F8FFF8'))
    layout['xaxis'] = dict(
        title=xaxis_title, gridcolor='rgba(76,175,80,0.12)',
        linecolor='rgba(76,175,80,0.3)', tickfont=dict(color='#B0BEC5'))
    layout['yaxis'] = dict(
        title=yaxis_title, gridcolor='rgba(76,175,80,0.12)',
        linecolor='rgba(76,175,80,0.3)', tickfont=dict(color='#B0BEC5'))
    return layout

# ── Paleta ────────────────────────────────────────────────────────────
C = {
    'verde':'#1B4D2E','vmed':'#2E7D32','vvivo':'#4CAF50',
    'vcl':'#A5D6A7','dorado':'#FFD700','dosc':'#F57F17',
    'azul':'#00BCD4','rojo':'#FF5252','naran':'#FF6D00',
    'purp':'#CE93D8','blanco':'#F8FFF8','gris':'#B0BEC5',
    'negro':'#060E08','oscuro':'#0A1F0E','teal':'#00897B',
}

# ── Motor MILP ────────────────────────────────────────────────────────
try:
    from milp_core import resolver_milp, SD_DEFAULT, TEC_ELEG, PRODUCTOS
    MILP_OK = True
except Exception:
    MILP_OK = False

# ── FO5 ───────────────────────────────────────────────────────────────
try:
    from FO5_capex import (resolver_FO5, CAPEX_BASE, SITIOS,
                            WACC, HORIZONTE, PRESUPUESTO,
                            PRECIO_CARBONO, FACTOR_INCENTIVO)
    FO5_OK = True
except Exception:
    FO5_OK = False
    CAPEX_BASE = {
        'molienda':1_200_000,'secado':1_800_000,'compostaje':850_000,
        'fermentacion':6_500_000,'transesterificacion':3_200_000,
        'extraccion_solventes':12_000_000,'hidrolisis_enzimatica':8_500_000,
        'pirolisis':5_500_000,'carbonizacion':4_800_000,
    }
    SITIOS = {
        'Apartado': {'costo_terreno_usd_ha':170_000,'area_necesaria_ha':5,
                     'dist_campo_km':18,'costo_infra_usd':500_000,
                     'acceso_vial':5,'latitud':7.8839,'longitud':-76.6275,
                     'costo_total_sitio':1_350_000,'costo_log_anual':2_507_166,
                     'descripcion':'Centro logístico consolidado, Zona Franca'},
        'Turbo':    {'costo_terreno_usd_ha':65_000,'area_necesaria_ha':5,
                     'dist_campo_km':35,'costo_infra_usd':350_000,
                     'acceso_vial':4,'latitud':8.0968,'longitud':-76.7291,
                     'costo_total_sitio':675_000,'costo_log_anual':4_876_683,
                     'descripcion':'Puerto Antioquia, exportación directa'},
        'Carepa':   {'costo_terreno_usd_ha':55_000,'area_necesaria_ha':5,
                     'dist_campo_km':22,'costo_infra_usd':420_000,
                     'acceso_vial':4,'latitud':7.7617,'longitud':-76.6566,
                     'costo_total_sitio':695_000,'costo_log_anual':3_047_676,
                     'descripcion':'Zona central Urabá, equidistante'},
        'Chigorodo':{'costo_terreno_usd_ha':48_000,'area_necesaria_ha':5,
                     'dist_campo_km':28,'costo_infra_usd':380_000,
                     'acceso_vial':3,'latitud':7.6719,'longitud':-76.6852,
                     'costo_total_sitio':620_000,'costo_log_anual':3_904_866,
                     'descripcion':'Menor costo terreno, buen acceso sur'},
        'Mutata':   {'costo_terreno_usd_ha':32_000,'area_necesaria_ha':5,
                     'dist_campo_km':42,'costo_infra_usd':550_000,
                     'acceso_vial':2,'latitud':7.2444,'longitud':-76.4361,
                     'costo_total_sitio':710_000,'costo_log_anual':5_857_119,
                     'descripcion':'Menor CAPEX terreno, mayor costo logístico'},
    }
    WACC=0.125; HORIZONTE=20; PRESUPUESTO=40_000_000
    PRECIO_CARBONO=18.0; FACTOR_INCENTIVO=0.725

# ── Datos ─────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    dfs = {}
    for nombre, ruta in [
        ('pareto',    'data/frente_pareto_5D.csv'),
        ('pareto_4D', 'data/frente_pareto.csv'),
        ('sd',        'data/datos_vensim.csv'),
        ('resumen',   'data/resumen_sd.csv'),
    ]:
        try:
            dfs[nombre] = pd.read_csv(ruta)
        except Exception:
            dfs[nombre] = None
    return dfs

datos = cargar_datos()

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-text">🌿 BIOREFINERÍA</div>
      <div class="sidebar-logo-text" style="color:#4CAF50!important">URABÁ</div>
      <div class="sidebar-sub">SD-MILP 5D · Grupo ALIADO</div>
      <div class="sidebar-sub">Universidad de Antioquia · 2025</div>
      <div style="margin-top:0.5rem;">
        <span style="background:rgba(255,109,0,0.2);border:1px solid #FF6D00;
          color:#FF6D00;padding:0.15rem 0.6rem;border-radius:10px;
          font-family:Space Mono;font-size:0.6rem;font-weight:700;">
          VERSIÓN 5D — FO1·FO2·FO3·FO4·FO5
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio("", [
        "🏠  Dashboard Ejecutivo",
        "⚙️  Optimizador MILP",
        "📊  Explorador Pareto 5D",
        "💰  Análisis CAPEX",
        "🌱  Dinámica SD",
    ], label_visibility='collapsed')

    st.markdown("---")
    st.markdown('<p style="font-family:Space Mono;font-size:0.65rem;color:#FFD700;'
                'letter-spacing:0.12em;text-transform:uppercase;">Parámetros SD</p>',
                unsafe_allow_html=True)

    eta = st.slider("η cadena logística", 0.20, 0.90, 0.42, 0.01)
    sup = st.number_input("Superficie (Ha)", 10000, 60000, 36932, 1000)

    q_gen   = sup * 3.4375 * 12
    q_total = q_gen * eta

    st.markdown(f"""
    <div style="background:rgba(27,77,46,0.3);border:1px solid rgba(76,175,80,0.3);
                border-radius:8px;padding:0.8rem;margin-top:0.5rem;">
      <div style="font-family:Space Mono;font-size:0.6rem;color:#B0BEC5;
                  text-transform:uppercase;margin-bottom:0.4rem;">Biomasa</div>
      <div style="font-family:Syne;font-size:1rem;font-weight:800;color:#FFD700;">
        {q_gen/1e6:.2f}M Ton/año</div>
      <div style="font-family:DM Sans;font-size:0.75rem;color:#A5D6A7;">
        Recolectada: {q_total/1e6:.3f}M · η={eta:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-family:Space Mono;font-size:0.55rem;color:#546E7A;'
                'text-align:center;">Juan Carlos Gaviria Chaverra<br>'
                'jcarlos.gaviria@udea.edu.co</p>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 1 — DASHBOARD EJECUTIVO (idéntico a v1 + badge v2)
# ═════════════════════════════════════════════════════════════════════
if '🏠' in pagina:

    st.markdown(f"""
    <div class="hero-header">
      <div class="hero-title">BIOREFINERÍA INTEGRAL<br>CADENA BANANERA URABÁ
        <span class="v2-badge">VERSIÓN 5D</span>
      </div>
      <div class="hero-subtitle">
        Modelo Híbrido SD-MILP · Optimización Multiobjetivo 5D · FO1-FO5
      </div>
      <div style="margin-top:0.8rem;">
        <span class="hero-badge">🌿 AUGURA 2024</span>
        <span class="hero-badge">📊 Pareto 5D</span>
        <span class="hero-badge">⚙️ 10 Tecnologías</span>
        <span class="hero-badge">🏭 16 Productos</span>
        <span class="hero-badge">💰 CAPEX + Localización</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs 5D
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        (k1,'#4CAF50','FO1 · UTILIDAD','USD 356.5M','/año','Ingreso bruto: USD 436.3M'),
        (k2,'#00BCD4','FO2 · GEI NETO','-18,861','tCO₂/año','🌿 Carbono negativo'),
        (k3,'#FFD700','FO3 · EMPLEO','74,991','emp/año','Directo+Indirecto'),
        (k4,'#FF6D00','FO4 · APROVECH.','42%→100%','α_BR','η=0.42'),
        (k5,'#CE93D8','FO5 · CAPEX','USD ~25M','inversión','Datos simulados'),
    ]
    for col,color,label,val,unit,sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{color};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{val}</div>
              <div class="kpi-delta">{unit}</div>
              <div style="font-size:0.7rem;color:#546E7A;margin-top:0.3rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # Radar 5D
    st.markdown('<div class="section-title">Solución de Compromiso 5D</div>',
                unsafe_allow_html=True)
    c_left, c_right = st.columns([3,2])

    with c_left:
        cats  = ['FO1 Utilidad','FO2 GEI','FO3 Empleo',
                 'FO4 Aprovech.','FO5 CAPEX']
        vals  = [0.64, 0.393, 0.69, 1.0, 0.60]
        colors_r = ['#4CAF50','#00BCD4','#FFD700','#FF6D00','#CE93D8']

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(76,175,80,0.12)',
            line=dict(color='#FFD700', width=3),
            marker=dict(size=8, color='#FFD700',
                        line=dict(color='#060E08', width=2)),
            name='Compromiso L2 5D'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[1,1,1,1,1,1], theta=cats+[cats[0]],
            line=dict(color='rgba(76,175,80,0.3)', width=1, dash='dot'),
            fill='none', name='Utópico 5D',
        ))
        fig_radar.update_layout(
            paper_bgcolor='rgba(6,14,8,0.0)',
            plot_bgcolor='rgba(10,31,14,0.4)',
            font=dict(family='DM Sans', color='#F8FFF8', size=12),
            polar=dict(
                bgcolor='rgba(10,31,14,0.6)',
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='rgba(76,175,80,0.2)',
                                tickfont=dict(color='#546E7A', size=9)),
                angularaxis=dict(tickfont=dict(color='#F8FFF8', size=11,
                                               family='Space Mono')),
            ),
            height=340,
            title=dict(text='Perfil Compromiso L2 — 5 Dimensiones', x=0.5,
                       font=dict(family='Syne', size=13, color='#F8FFF8')),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_right:
        st.markdown("""
        <div style="background:rgba(10,31,14,0.8);border:1px solid rgba(76,175,80,0.3);
                    border-left:4px solid #FFD700;border-radius:10px;padding:1.2rem;">
          <div style="font-family:Space Mono;font-size:0.6rem;color:#FFD700;
                      letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;">
            Solución compromiso L2 — 5D
          </div>
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;padding:0.4rem 0;">
              FO1 Utilidad</td>
              <td style="font-family:Syne;font-size:0.9rem;font-weight:700;color:#4CAF50;text-align:right;">
              USD 229.3M/año</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.15);">
              <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;padding:0.4rem 0;">
              FO2 GEI neto</td>
              <td style="font-family:Syne;font-size:0.9rem;font-weight:700;color:#00BCD4;text-align:right;">
              +9,948 tCO₂/año</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.15);">
              <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;padding:0.4rem 0;">
              FO3 Empleo</td>
              <td style="font-family:Syne;font-size:0.9rem;font-weight:700;color:#FFD700;text-align:right;">
              13,462 emp/año</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.15);">
              <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;padding:0.4rem 0;">
              FO4 Aprovech.</td>
              <td style="font-family:Syne;font-size:0.9rem;font-weight:700;color:#FF6D00;text-align:right;">
              100% α_BR</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.15);">
              <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;padding:0.4rem 0;">
              FO5 CAPEX</td>
              <td style="font-family:Syne;font-size:0.9rem;font-weight:700;color:#CE93D8;text-align:right;">
              USD ~25M</td></tr>
            <tr style="border-top:1px solid rgba(255,215,0,0.3);">
              <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;padding:0.4rem 0;">
              Dist. utópico 5D</td>
              <td style="font-family:Syne;font-size:0.9rem;font-weight:700;color:#FFD700;text-align:right;">
              ~0.72</td></tr>
          </table>
          <div style="margin-top:1rem;padding-top:0.8rem;
                      border-top:1px solid rgba(76,175,80,0.2);">
            <div style="font-family:Space Mono;font-size:0.6rem;color:#546E7A;
                        text-transform:uppercase;letter-spacing:0.1em;">
              Datos FO5 simulados — escenario referencia</div>
            <div style="font-family:DM Sans;font-size:0.75rem;color:#FF6D00;
                        margin-top:0.3rem;">
              ⚠️ Actualizar con datos reales de campo</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Sankey biomasa
    st.markdown('<div class="section-title">Flujo de Biomasa — Cadena Bananera</div>',
                unsafe_allow_html=True)
    per = q_gen - q_total
    fig_sankey = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20, thickness=25,
            label=["Campo\n36,932 Ha","Biomasa\nCampo","Packing\nPlant",
                   "Red\nLogística","Pérdida\nRed","Biorefinería",
                   "Biochar","Compost","Bioenergía","Extractos","Fibras"],
            color=[C['vmed'],C['vvivo'],C['dosc'],C['naran'],C['rojo'],
                   C['azul'],C['purp'],C['verde'],C['azul'],'#00897B',C['naran']],
            line=dict(color='rgba(6,14,8,0.8)', width=1),
        ),
        link=dict(
            source=[0,0,1,2,3,3,5,5,5,5,5],
            target=[1,2,3,3,4,5,6,7,8,9,10],
            value=[q_gen*0.873/1e6,q_gen*0.127/1e6,
                   q_gen*0.873/1e6,q_gen*0.127/1e6,
                   per/1e6,q_total/1e6,
                   q_total*0.12/1e6,q_total*0.35/1e6,
                   q_total*0.30/1e6,q_total*0.08/1e6,q_total*0.15/1e6],
            color=['rgba(76,175,80,0.35)']*5+['rgba(0,188,212,0.35)']+
                  ['rgba(206,147,216,0.4)','rgba(27,77,46,0.4)',
                   'rgba(0,188,212,0.4)','rgba(0,137,123,0.4)',
                   'rgba(255,109,0,0.4)'],
        )
    ))
    fig_sankey.update_layout(
        paper_bgcolor='rgba(6,14,8,0.0)',
        plot_bgcolor='rgba(10,31,14,0.4)',
        font=dict(family='DM Sans', color='#F8FFF8', size=12),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        title=dict(
            text=f'Flujo biomasa · η={eta:.0%} · {q_gen/1e6:.2f}M Ton/año',
            x=0.5, font=dict(family='Syne', size=14, color='#F8FFF8')),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 2 — OPTIMIZADOR MILP (idéntico a v1)
# ═════════════════════════════════════════════════════════════════════
elif '⚙️' in pagina:
    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">OPTIMIZADOR MILP<br>EN TIEMPO REAL</div>
      <div class="hero-subtitle">FO1-FO4 · Ajusta parámetros · Ejecuta · Visualiza</div>
    </div>
    """, unsafe_allow_html=True)

    if not MILP_OK:
        st.warning("Motor MILP no disponible — verifica milp_core.py")
    else:
        col_p, col_r = st.columns([1,2], gap='large')
        with col_p:
            st.markdown('<div class="section-title">Configuración</div>',
                        unsafe_allow_html=True)
            objetivo = st.selectbox("Función objetivo", [
                "FO1 — Maximizar Utilidad",
                "FO2 — Minimizar GEI",
                "FO3 — Maximizar Empleo",
                "FO4 — Maximizar Aprovechamiento",
                "Compromiso — Suma Ponderada",
            ])
            obj_map = {
                "FO1 — Maximizar Utilidad":       "FO1",
                "FO2 — Minimizar GEI":            "FO2",
                "FO3 — Maximizar Empleo":         "FO3",
                "FO4 — Maximizar Aprovechamiento":"FO4",
                "Compromiso — Suma Ponderada":    "compromiso",
            }
            obj_key = obj_map[objetivo]
            precio_factor = st.slider("Factor de precios", 0.5, 2.0, 1.0, 0.05)
            gei_factor    = st.slider("Factor emisiones", 0.5, 2.0, 1.0, 0.05)
            phi = st.slider("φ biochar (tCO₂/ton)", 0.5, 3.5, 1.65, 0.05)
            mu  = st.slider("μ empleo indirecto", 1.0, 5.0, 2.5, 0.1)
            w_list = None
            if obj_key == 'compromiso':
                w1=st.slider("w₁ Utilidad",0.0,1.0,0.25,0.05)
                w2=st.slider("w₂ GEI",0.0,1.0,0.25,0.05)
                w3=st.slider("w₃ Empleo",0.0,1.0,0.25,0.05)
                w4=st.slider("w₄ Aprovech.",0.0,1.0,0.25,0.05)
                w_list=[w1,w2,w3,w4]
            sd_custom={'Q_total_anual':q_total,'Q_gen_anual':q_gen,
                       'eta_cadena':eta,'GEI_base':7796.6,
                       'fertilidad':0.449,'superficie':sup}
            correr=st.button("🚀 EJECUTAR OPTIMIZACIÓN",
                             use_container_width=True, type='primary')

        with col_r:
            if correr:
                with st.spinner("⚡ Resolviendo..."):
                    t0=time.time()
                    res=resolver_milp(objetivo=obj_key,sd_params=sd_custom,
                                      eta=eta,precio_factor=precio_factor,
                                      gei_factor=gei_factor,phi=phi,mu=mu,w=w_list)
                    dt=time.time()-t0
                if res.get('error'):
                    st.error("❌ Modelo infactible.")
                else:
                    st.success(f"✅ Solución en {dt:.1f}s")
                    k1,k2,k3,k4=st.columns(4)
                    for col,color,lbl,val in [
                        (k1,'#4CAF50','Utilidad',f"USD {res['FO1']/1e6:.1f}M/año"),
                        (k2,'#00BCD4','GEI neto',f"{res['FO2']:,.0f} tCO₂/año"),
                        (k3,'#FFD700','Empleo',f"{res['emp_total']:,.0f} emp/año"),
                        (k4,'#FF6D00','Aprovech.',f"{res['alpha_BR']*100:.1f}% α_BR"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class="kpi-card" style="border-top-color:{color};">
                              <div class="kpi-label">{lbl}</div>
                              <div class="kpi-value" style="color:{color};
                                   font-size:1.3rem;">{val}</div>
                            </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="height:300px;display:flex;align-items:center;
                            justify-content:center;background:rgba(10,31,14,0.4);
                            border:1px dashed rgba(76,175,80,0.3);border-radius:12px;">
                  <div style="text-align:center;">
                    <div style="font-size:3rem;">⚙️</div>
                    <div style="font-family:Syne;font-size:1.2rem;
                                font-weight:700;color:#4CAF50;margin:0.5rem 0;">
                      Configura y ejecuta</div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 3 — EXPLORADOR PARETO 5D
# ═════════════════════════════════════════════════════════════════════
elif '📊' in pagina:
    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">EXPLORADOR<br>FRENTE DE PARETO 5D</div>
      <div class="hero-subtitle">FO1·FO2·FO3·FO4·FO5 · 3 métodos · Espacio 5D</div>
    </div>
    """, unsafe_allow_html=True)

    df_p5 = datos.get('pareto')
    df_p4 = datos.get('pareto_4D')

    tab_5d, tab_4d = st.tabs(["📊 Pareto 5D (con CAPEX)", "📈 Pareto 4D (referencia)"])

    def mostrar_pareto(df_p, sufijo=''):
        if df_p is None:
            st.markdown("""
            <div style="background:rgba(255,109,0,0.1);border:1px solid #FF6D00;
                        border-radius:10px;padding:1.5rem;text-align:center;">
              <div style="font-family:Syne;font-size:1rem;font-weight:700;color:#FF6D00;">
                ⚠️ CSV no encontrado</div>
              <div style="font-family:DM Sans;font-size:0.85rem;color:#B0BEC5;
                          margin-top:0.5rem;">
                Ejecuta multiobjetivo_pareto_v2.py en Colab y sube el CSV
              </div>
            </div>""", unsafe_allow_html=True)
            return

        # Scatter FO1 vs FO2
        g1, g2 = st.columns(2)
        with g1:
            fo_x = 'FO1_kUSD' if 'FO1_kUSD' in df_p.columns else df_p.columns[0]
            fo_y = 'FO2_tCO2' if 'FO2_tCO2' in df_p.columns else df_p.columns[1]
            fig_sc = go.Figure()
            fig_sc.add_hline(y=0, line_dash='dash',
                             line_color='rgba(255,82,82,0.4)',
                             annotation_text='Carbono neutro',
                             annotation_font=dict(color='#FF5252', size=9))
            if 'metodo' in df_p.columns:
                for met, col_m in [('suma_ponderada','#4CAF50'),
                                    ('chebyshev','#00BCD4'),
                                    ('eps','#FFD700')]:
                    sub = df_p[df_p['metodo'].str.startswith(met)]
                    if len(sub):
                        fig_sc.add_trace(go.Scatter(
                            x=sub[fo_x], y=sub[fo_y], mode='markers',
                            marker=dict(size=10, color=col_m,
                                        line=dict(color='#060E08', width=1),
                                        opacity=0.85),
                            name=met,
                        ))
            else:
                fig_sc.add_trace(go.Scatter(
                    x=df_p[fo_x], y=df_p[fo_y], mode='markers',
                    marker=dict(size=10, color='#4CAF50')))
            fig_sc.update_layout(
                **layout_base(height=380,
                               title_text=f'FO1 vs FO2{sufijo}',
                               xaxis_title='Utilidad (kUSD/año)',
                               yaxis_title='GEI neto (tCO₂/año)',
                               showlegend=True),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with g2:
            # Coordenadas paralelas
            cols_norm = [c for c in ['FO1_norm','FO2_norm','FO3_norm',
                                      'FO4_norm','FO5_norm']
                         if c in df_p.columns]
            cols_lbl  = ['FO1\nUtilidad','FO2\nGEI','FO3\nEmpleo',
                         'FO4\nAprovech.','FO5\nCAPEX'][:len(cols_norm)]

            if cols_norm and 'dist_utopico' in df_p.columns:
                dist = df_p['dist_utopico']
                dims = [dict(label=l, values=df_p[c], range=[0,1])
                        for l, c in zip(cols_lbl, cols_norm)]
                fig_cp = go.Figure(go.Parcoords(
                    line=dict(color=dist,
                              colorscale=[[0,'#FFD700'],[0.5,'#4CAF50'],[1,'#1B4D2E']],
                              showscale=True,
                              cmin=float(dist.min()), cmax=float(dist.max()),
                              colorbar=dict(thickness=10, len=0.7,
                                            title=dict(text='Dist.\nutópico',
                                                       font=dict(color='#B0BEC5',size=9)),
                                            tickfont=dict(color='#B0BEC5',size=8))),
                    dimensions=dims,
                    labelfont=dict(color='#F8FFF8', size=11, family='Space Mono'),
                    tickfont=dict(color='#546E7A', size=8),
                ))
                fig_cp.update_layout(
                    paper_bgcolor='rgba(6,14,8,0.0)',
                    plot_bgcolor='rgba(10,31,14,0.4)',
                    font=dict(family='DM Sans', color='#F8FFF8'),
                    height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                    title=dict(
                        text=f'Coordenadas Paralelas{sufijo}', x=0.5,
                        font=dict(family='Syne', size=14, color='#F8FFF8')),
                )
                st.plotly_chart(fig_cp, use_container_width=True)

        # Tabla
        st.dataframe(df_p.round(2).reset_index(drop=True),
                     use_container_width=True, height=220)
        csv = df_p.to_csv(index=False).encode('utf-8')
        st.download_button(f"⬇️ Descargar CSV{sufijo}", csv,
                           f"pareto{sufijo}.csv", "text/csv")

    with tab_5d:
        mostrar_pareto(df_p5, ' 5D')
    with tab_4d:
        mostrar_pareto(df_p4, ' 4D (referencia)')


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 4 — ANÁLISIS CAPEX ← NUEVA en v2
# ═════════════════════════════════════════════════════════════════════
elif '💰' in pagina:

    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">ANÁLISIS CAPEX<br>Y LOCALIZACIÓN ÓPTIMA</div>
      <div class="hero-subtitle">
        FO5 · Inversión · Sitios candidatos · VPN · TIR · PRI · ROCE
      </div>
      <div style="margin-top:0.8rem;">
        <span class="hero-badge" style="border-color:#FF6D00;color:#FF6D00;">
          ⚠️ Datos simulados — escenario de referencia</span>
        <span class="hero-badge">WACC 12.5%</span>
        <span class="hero-badge">Horizonte 20 años</span>
        <span class="hero-badge">Precio carbono USD 18/tCO₂</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_capex, tab_sitios, tab_financiero = st.tabs([
        "🏭 CAPEX por Tecnología",
        "📍 Comparación de Sitios",
        "📈 Indicadores Financieros",
    ])

    # ── TAB 1: CAPEX por tecnología ───────────────────────────────────
    with tab_capex:
        st.markdown('<div class="section-title">CAPEX por Tecnología (datos simulados)</div>',
                    unsafe_allow_html=True)

        capex_df = pd.DataFrame([
            {'Tecnología': t.replace('_',' ').title(),
             'CAPEX (USD)': v,
             'CAPEX (MUSD)': round(v/1e6, 2),
             'Categoría': ('Pretratamiento' if t in ['molienda','secado']
                           else 'Biológico' if t in ['compostaje','fermentacion']
                           else 'Química fina' if t in ['transesterificacion',
                                                        'extraccion_solventes',
                                                        'hidrolisis_enzimatica']
                           else 'Termoquímica')}
            for t, v in CAPEX_BASE.items() if v > 0
        ]).sort_values('CAPEX (USD)', ascending=True)

        colores_cat = {
            'Pretratamiento': '#4CAF50',
            'Biológico':      '#00BCD4',
            'Química fina':   '#FFD700',
            'Termoquímica':   '#CE93D8',
        }

        fig_capex = go.Figure()
        for cat, col_c in colores_cat.items():
            sub = capex_df[capex_df['Categoría']==cat]
            if len(sub):
                fig_capex.add_trace(go.Bar(
                    x=sub['CAPEX (MUSD)'],
                    y=sub['Tecnología'],
                    orientation='h',
                    name=cat,
                    marker=dict(color=col_c,
                                line=dict(color='rgba(6,14,8,0.5)', width=0.5)),
                    text=[f"USD {v:.2f}M" for v in sub['CAPEX (MUSD)']],
                    textposition='outside',
                    textfont=dict(color='#B0BEC5', size=10, family='Space Mono'),
                ))

        fig_capex.update_layout(
            **layout_base(height=400,
                           title_text='CAPEX por Tecnología — Datos Simulados Urabá',
                           xaxis_title='CAPEX (Millones USD)',
                           showlegend=True),
            barmode='stack',
        )
        st.plotly_chart(fig_capex, use_container_width=True)

        # Tabla CAPEX
        st.markdown('<div class="section-title">Detalle CAPEX</div>',
                    unsafe_allow_html=True)
        capex_show = capex_df[['Tecnología','Categoría',
                                'CAPEX (MUSD)']].reset_index(drop=True)
        st.dataframe(capex_show, use_container_width=True, height=300)

        # Nota
        st.markdown("""
        <div style="background:rgba(255,109,0,0.1);border:1px solid #FF6D00;
                    border-left:4px solid #FF6D00;border-radius:8px;padding:1rem;
                    margin-top:1rem;">
          <div style="font-family:Space Mono;font-size:0.65rem;color:#FF6D00;
                      text-transform:uppercase;letter-spacing:0.1em;">
            Nota sobre los datos</div>
          <div style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;
                      margin-top:0.4rem;line-height:1.6;">
            Los valores de CAPEX son estimaciones simuladas basadas en literatura
            latinoamericana y datos de FENOGE 2024. Serán reemplazados por cotizaciones
            reales de ingeniería de detalle cuando estén disponibles.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Comparación de sitios ──────────────────────────────────
    with tab_sitios:
        st.markdown('<div class="section-title">Sitios Candidatos — Región Urabá</div>',
                    unsafe_allow_html=True)

        # Mapa interactivo
        sitios_df = pd.DataFrame([
            {'Municipio': s,
             'lat': d['latitud'],
             'lon': d['longitud'],
             'Costo terreno (USD/Ha)': d['costo_terreno_usd_ha'],
             'Distancia campo (km)': d['dist_campo_km'],
             'Acceso vial': d['acceso_vial'],
             'Costo total sitio (USD)': d['costo_total_sitio'],
             'Costo log. anual (USD)': d['costo_log_anual'],
             'Descripción': d['descripcion'],
             'size': d['acceso_vial']*8,
            }
            for s, d in SITIOS.items()
        ])

        fig_mapa = go.Figure()

        colores_sitio = {
            'Apartado':  '#FFD700',
            'Turbo':     '#4CAF50',
            'Carepa':    '#00BCD4',
            'Chigorodo': '#CE93D8',
            'Mutata':    '#FF6D00',
        }

        for _, row in sitios_df.iterrows():
            fig_mapa.add_trace(go.Scattermapbox(
                lat=[row['lat']], lon=[row['lon']],
                mode='markers+text',
                marker=dict(
                    size=row['size'],
                    color=colores_sitio.get(row['Municipio'], '#4CAF50'),
                    opacity=0.9,
                ),
                text=[row['Municipio']],
                textposition='top right',
                textfont=dict(color='white', size=12, family='Syne'),
                name=row['Municipio'],
                hovertemplate=(
                    f"<b>{row['Municipio']}</b><br>"
                    f"Costo terreno: USD {row['Costo terreno (USD/Ha)']:,}/Ha<br>"
                    f"Dist. campo: {row['Distancia campo (km)']} km<br>"
                    f"Costo sitio: USD {row['Costo total sitio (USD)']:,}<br>"
                    f"Acceso vial: {row['Acceso vial']}/5<br>"
                    f"{row['Descripción']}<extra></extra>"
                ),
            ))

        fig_mapa.update_layout(
            mapbox=dict(
                style='carto-darkmatter',
                center=dict(lat=7.85, lon=-76.65),
                zoom=8.5,
            ),
            paper_bgcolor='rgba(6,14,8,0.0)',
            height=450,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(text='Mapa de sitios candidatos — Región Urabá, Antioquia',
                       x=0.5, font=dict(family='Syne', size=14, color='#F8FFF8')),
            legend=dict(bgcolor='rgba(6,14,8,0.8)',
                        bordercolor='rgba(76,175,80,0.3)',
                        font=dict(color='#F8FFF8')),
        )
        st.plotly_chart(fig_mapa, use_container_width=True)

        # Comparación en tarjetas
        st.markdown('<div class="section-title">Comparación detallada por municipio</div>',
                    unsafe_allow_html=True)

        cols_sitios = st.columns(len(SITIOS))
        mejor_sitio = 'Apartado'  # según FO5 con datos simulados

        for i, (nombre, datos_s) in enumerate(SITIOS.items()):
            es_optimo = nombre == mejor_sitio
            with cols_sitios[i]:
                clase = 'sitio-optimo' if es_optimo else ''
                badge = '⭐ ÓPTIMO' if es_optimo else ''
                color_n = '#FFD700' if es_optimo else colores_sitio.get(nombre,'#4CAF50')
                st.markdown(f"""
                <div class="sitio-card {clase}">
                  <div style="font-family:Syne;font-size:0.95rem;font-weight:800;
                              color:{color_n};margin-bottom:0.5rem;">
                    {nombre} {badge}</div>
                  <div style="font-family:Space Mono;font-size:0.65rem;
                              color:#B0BEC5;line-height:1.8;">
                    Terreno: USD {datos_s['costo_terreno_usd_ha']:,}/Ha<br>
                    Dist.: {datos_s['dist_campo_km']} km<br>
                    Costo sitio: USD {datos_s['costo_total_sitio']/1e3:.0f}k<br>
                    Log. anual: USD {datos_s['costo_log_anual']/1e6:.2f}M<br>
                    Acceso: {'⭐'*datos_s['acceso_vial']}
                  </div>
                  <div style="font-family:DM Sans;font-size:0.7rem;color:#546E7A;
                              margin-top:0.5rem;">{datos_s['descripcion']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Gráfico de radar de sitios
        st.markdown('<div class="section-title">Radar comparativo de sitios</div>',
                    unsafe_allow_html=True)

        categorias_r = ['Costo bajo', 'Dist. corta', 'Acceso vial',
                         'Log. barata', 'Total conveniente']
        fig_rad_s = go.Figure()

        max_costo = max(d['costo_total_sitio'] for d in SITIOS.values())
        max_dist  = max(d['dist_campo_km'] for d in SITIOS.values())
        max_log   = max(d['costo_log_anual'] for d in SITIOS.values())

        for nombre, datos_s in SITIOS.items():
            v_costo = 1 - datos_s['costo_total_sitio']/max_costo
            v_dist  = 1 - datos_s['dist_campo_km']/max_dist
            v_vial  = datos_s['acceso_vial']/5
            v_log   = 1 - datos_s['costo_log_anual']/max_log
            v_total = (v_costo+v_dist+v_vial+v_log)/4
            vals_r  = [v_costo, v_dist, v_vial, v_log, v_total]

            color_sitio = colores_sitio.get(nombre,'#4CAF50')
            hex_c = color_sitio.lstrip('#')
            r_c,g_c,b_c = int(hex_c[0:2],16),int(hex_c[2:4],16),int(hex_c[4:6],16)
            fig_rad_s.add_trace(go.Scatterpolar(
                r=vals_r+[vals_r[0]], theta=categorias_r+[categorias_r[0]],
                fill='toself', name=nombre,
                line=dict(color=color_sitio, width=2),
                fillcolor=f'rgba({r_c},{g_c},{b_c},0.12)',
            ))

        fig_rad_s.update_layout(
            paper_bgcolor='rgba(6,14,8,0.0)',
            plot_bgcolor='rgba(10,31,14,0.4)',
            font=dict(family='DM Sans', color='#F8FFF8', size=12),
            polar=dict(
                bgcolor='rgba(10,31,14,0.6)',
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='rgba(76,175,80,0.2)',
                                tickfont=dict(color='#546E7A',size=9)),
                angularaxis=dict(tickfont=dict(color='#F8FFF8',size=10,
                                               family='Space Mono')),
            ),
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
            title=dict(text='Perfil comparativo de sitios candidatos', x=0.5,
                       font=dict(family='Syne', size=14, color='#F8FFF8')),
            showlegend=True,
            legend=dict(bgcolor='rgba(6,14,8,0.7)',
                        bordercolor='rgba(76,175,80,0.3)',
                        borderwidth=1, font=dict(color='#F8FFF8')),
        )
        st.plotly_chart(fig_rad_s, use_container_width=True)

    # ── TAB 3: Indicadores financieros ────────────────────────────────
    with tab_financiero:
        st.markdown('<div class="section-title">Análisis Financiero — Escenario Referencia</div>',
                    unsafe_allow_html=True)

        # Sliders parámetros financieros
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            wacc_s = st.slider("WACC (%)", 8.0, 20.0, 12.5, 0.5) / 100
        with cf2:
            horiz_s = st.slider("Horizonte (años)", 10, 30, 20, 1)
        with cf3:
            precio_c = st.slider("Precio carbono (USD/tCO₂)", 5.0, 50.0, 18.0, 1.0)

        # CAPEX simulado de la solución compromiso
        tec_compromiso = ['molienda','secado','compostaje',
                          'extraccion_solventes','pirolisis','carbonizacion']
        capex_comp = sum(CAPEX_BASE.get(t,0) for t in tec_compromiso)
        costo_sit  = SITIOS['Apartado']['costo_total_sitio']
        capex_total= capex_comp + costo_sit
        capex_neto = capex_total * FACTOR_INCENTIVO

        # Utilidad neta anual (de la solución compromiso L2)
        utilidad_anual = 229_300_000
        biochar_anual  = 45_350
        ingreso_carbono= precio_c * 1.65 * biochar_anual

        # VPN
        flujos = [(utilidad_anual + ingreso_carbono) / (1+wacc_s)**t
                  for t in range(1, horiz_s+1)]
        vpn = sum(flujos) - capex_neto

        # TIR
        def npv_r(r):
            return sum((utilidad_anual+ingreso_carbono)/(1+r)**t
                       for t in range(1, horiz_s+1)) - capex_neto
        try:
            lo, hi = 0.001, 5.0
            for _ in range(60):
                mid = (lo+hi)/2
                if npv_r(mid) > 0: lo = mid
                else: hi = mid
            tir = (lo+hi)/2
        except Exception:
            tir = 0.0

        # PRI
        acum, pri = 0, horiz_s
        for t in range(1, horiz_s+1):
            acum += utilidad_anual + ingreso_carbono
            if acum >= capex_neto:
                pri = t; break

        # ROCE
        roce = (utilidad_anual / max(capex_total, 1)) * 100

        # KPIs financieros
        f1,f2,f3,f4 = st.columns(4)
        for col, color, lbl, val, sub in [
            (f1,'#4CAF50','VPN',f"USD {vpn/1e6:.1f}M",
             f"WACC={wacc_s*100:.1f}% · {horiz_s} años"),
            (f2,'#FFD700','TIR',f"{tir*100:.1f}%",
             f"vs WACC {wacc_s*100:.1f}% → {'✅ viable' if tir>wacc_s else '❌ no viable'}"),
            (f3,'#00BCD4','PRI',f"{pri} años",
             f"Recuperación de USD {capex_neto/1e6:.1f}M netos"),
            (f4,'#CE93D8','ROCE',f"{roce:.1f}%",
             f"Retorno sobre USD {capex_total/1e6:.1f}M CAPEX total"),
        ]:
            with col:
                st.markdown(f"""
                <div class="capex-card" style="border-top-color:{color};">
                  <div class="kpi-label">{lbl}</div>
                  <div class="kpi-value" style="color:{color};font-size:1.5rem;">
                    {val}</div>
                  <div class="kpi-delta">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(10,31,14,0.6);border:1px solid rgba(76,175,80,0.3);
                    border-radius:10px;padding:1rem;margin-top:0.5rem;">
          <div style="font-family:Space Mono;font-size:0.6rem;color:#FFD700;
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
            Desglose CAPEX — Solución compromiso L2</div>
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;
                padding:0.3rem 0;">CAPEX tecnologías</td>
                <td style="font-family:Syne;font-size:0.9rem;font-weight:700;
                color:#4CAF50;text-align:right;">USD {capex_comp/1e6:.2f}M</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.1);">
                <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;
                padding:0.3rem 0;">Costo sitio (Apartadó)</td>
                <td style="font-family:Syne;font-size:0.9rem;font-weight:700;
                color:#00BCD4;text-align:right;">USD {costo_sit/1e6:.2f}M</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.1);">
                <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;
                padding:0.3rem 0;">CAPEX total bruto</td>
                <td style="font-family:Syne;font-size:0.9rem;font-weight:700;
                color:#FFD700;text-align:right;">USD {capex_total/1e6:.2f}M</td></tr>
            <tr style="border-top:1px solid rgba(255,215,0,0.3);">
                <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;
                padding:0.3rem 0;">CAPEX neto (con incentivos Ley 1715)</td>
                <td style="font-family:Syne;font-size:0.9rem;font-weight:700;
                color:#CE93D8;text-align:right;">USD {capex_neto/1e6:.2f}M</td></tr>
            <tr style="border-top:1px solid rgba(76,175,80,0.2);">
                <td style="font-family:DM Sans;font-size:0.8rem;color:#B0BEC5;
                padding:0.3rem 0;">Ingreso créditos carbono/año</td>
                <td style="font-family:Syne;font-size:0.9rem;font-weight:700;
                color:#4CAF50;text-align:right;">USD {ingreso_carbono/1e3:.0f}k/año</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Curva VPN vs WACC
        st.markdown('<div class="section-title">Sensibilidad VPN al WACC</div>',
                    unsafe_allow_html=True)
        waccs   = np.linspace(0.05, 0.25, 50)
        vpns    = [sum((utilidad_anual+ingreso_carbono)/(1+w)**t
                       for t in range(1,horiz_s+1)) - capex_neto
                   for w in waccs]

        fig_vpn = go.Figure()
        fig_vpn.add_trace(go.Scatter(
            x=waccs*100, y=[v/1e6 for v in vpns],
            mode='lines',
            line=dict(color='#4CAF50', width=3),
            fill='tozeroy',
            fillcolor='rgba(76,175,80,0.08)',
            name='VPN (MUSD)',
        ))
        fig_vpn.add_hline(y=0, line_dash='dash',
                          line_color='rgba(255,82,82,0.6)',
                          annotation_text='VPN = 0 (TIR)',
                          annotation_font=dict(color='#FF5252', size=10))
        fig_vpn.add_vline(x=wacc_s*100, line_dash='dot',
                          line_color='#FFD700',
                          annotation_text=f'WACC actual {wacc_s*100:.1f}%',
                          annotation_font=dict(color='#FFD700', size=10))
        fig_vpn.update_layout(
            **layout_base(height=350,
                           title_text='VPN vs WACC — Sensibilidad financiera',
                           xaxis_title='WACC (%)',
                           yaxis_title='VPN (Millones USD)'),
        )
        st.plotly_chart(fig_vpn, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 5 — DINÁMICA SD
# ═════════════════════════════════════════════════════════════════════
elif '🌱' in pagina:
    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">DINÁMICA DEL SISTEMA<br>MODELO VENSIM SD</div>
      <div class="hero-subtitle">
        Diagrama_Hibrido_Uraba_v6.mdl · 62 variables · 100 meses
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_sd = datos.get('sd')
    if df_sd is not None:
        vars_num = [c for c in df_sd.select_dtypes(include=[np.number]).columns
                    if c != 'mes']
        vars_sel = st.multiselect("Variables a graficar", vars_num,
                                  default=vars_num[:3] if len(vars_num)>=3
                                  else vars_num)
        if vars_sel:
            meses  = df_sd['mes'] if 'mes' in df_sd.columns else range(len(df_sd))
            colors = ['#4CAF50','#00BCD4','#FFD700','#FF6D00',
                      '#FF5252','#CE93D8','#80CBC4']
            fig_sd = go.Figure()
            for i, var in enumerate(vars_sel):
                if var in df_sd.columns:
                    fig_sd.add_trace(go.Scatter(
                        x=meses, y=df_sd[var], mode='lines',
                        name=var.replace('_',' '),
                        line=dict(color=colors[i%len(colors)], width=2.5),
                    ))
            fig_sd.update_layout(
                **layout_base(height=420,
                               title_text='Variables SD (Vensim)',
                               xaxis_title='Tiempo (meses)',
                               showlegend=True, hovermode='x unified'),
            )
            st.plotly_chart(fig_sd, use_container_width=True)
    else:
        st.info("Sube datos_vensim.csv a la carpeta data/ del repositorio.")

    # Simulación aproximada
    st.markdown('<div class="section-title">Simulación SD Interactiva</div>',
                unsafe_allow_html=True)
    m      = np.arange(0, 101)
    bio_d  = (q_gen/12/1000) * (1 + 0.0019*m/100*(1-m/200))
    rec_d  = bio_d * eta * (1 - np.exp(-m/5))
    gei_d  = (6800 + 1700*m/100)/1000

    fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sim.add_trace(go.Scatter(
        x=m, y=bio_d, name='Generación biomasa (kTon/mes)',
        line=dict(color='#4CAF50', width=3),
        fill='tozeroy', fillcolor='rgba(76,175,80,0.08)',
    ), secondary_y=False)
    fig_sim.add_trace(go.Scatter(
        x=m, y=rec_d, name='Recolección (kTon/mes)',
        line=dict(color='#00BCD4', width=2.5, dash='dash'),
    ), secondary_y=False)
    fig_sim.add_trace(go.Scatter(
        x=m, y=gei_d, name='GEI (kTonCO₂/mes)',
        line=dict(color='#FF5252', width=2, dash='dot'),
    ), secondary_y=True)
    fig_sim.update_layout(
        paper_bgcolor='rgba(6,14,8,0.0)',
        plot_bgcolor='rgba(10,31,14,0.4)',
        font=dict(family='DM Sans', color='#F8FFF8', size=12),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='x unified',
        showlegend=True,
        legend=dict(bgcolor='rgba(6,14,8,0.7)',
                    bordercolor='rgba(76,175,80,0.3)',
                    borderwidth=1, font=dict(color='#F8FFF8')),
        title=dict(
            text=f'Dinámica SD — η={eta:.0%} · {sup:,} Ha',
            x=0.5, font=dict(family='Syne', size=15, color='#F8FFF8')),
    )
    fig_sim.update_xaxes(title_text="Tiempo (meses)")
    fig_sim.update_yaxes(title_text="Biomasa (kTon/mes)", secondary_y=False)
    fig_sim.update_yaxes(title_text="GEI (kTonCO₂/mes)", secondary_y=True)
    st.plotly_chart(fig_sim, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1rem;
            border-top:1px solid rgba(76,175,80,0.2);text-align:center;">
  <span style="font-family:Space Mono;font-size:0.6rem;color:#546E7A;
               letter-spacing:0.1em;">
    BIOREFINERÍA INTEGRAL URABÁ · SD-MILP 5D · FO1-FO5 ·
    GRUPO ALIADO · UNIVERSIDAD DE ANTIOQUIA · 2025
    · <span style="color:#4CAF50;">36,932 Ha</span>
    · <span style="color:#FFD700;">1.265M Ton/año</span>
    · <span style="color:#CE93D8;">FO5 CAPEX + Localización</span>
    · <span style="color:#FF6D00;">Datos simulados — escenario referencia</span>
  </span>
</div>
""", unsafe_allow_html=True)
