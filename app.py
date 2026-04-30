import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.4rem;
    color: #3B82F6;
    margin-top: 1.5rem;
    border-bottom: 2px solid #E5E7EB;
    padding-bottom: 0.3rem;
}
.info-box {
    background-color: #F0F8FF;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card h3 { font-size: 1.8rem; margin: 0; }
.metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.9; }
.stage-card {
    background-color: #F8FAFC;
    padding: 0.5rem 0.8rem;
    border-radius: 5px;
    margin: 0.3rem 0;
    border-left: 3px solid #10B981;
    font-size: 0.9rem;
}
.warning-box {
    background-color: #FFF7ED;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #F97316;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================
class ProjectStage:
    """Model tahapan proyek konstruksi dengan distribusi triangular dan faktor risiko."""

    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic  = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors  = risk_factors  or {}
        self.dependencies  = dependencies  or []

    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        """Sampling durasi dengan distribusi triangular + faktor risiko."""
        base_duration = np.random.triangular(
            self.optimistic, self.most_likely, self.pessimistic, n_simulations
        )
        for _, rp in self.risk_factors.items():
            if rp['type'] == 'discrete':
                occurs = np.random.random(n_simulations) < rp['probability']
                base_duration = np.where(occurs, base_duration * (1 + rp['impact']), base_duration)
            elif rp['type'] == 'continuous':
                prod = np.random.normal(rp['mean'], rp['std'], n_simulations)
                base_duration = base_duration / np.clip(prod, 0.5, 1.5)
        return base_duration * risk_multiplier


class MonteCarloConstructionSimulation:
    """Simulasi Monte Carlo untuk proyek konstruksi gedung FITE."""

    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config   = stages_config
        self.num_simulations = num_simulations
        self.stages          = {}
        self.simulation_results = None
        self._initialize_stages()

    def _initialize_stages(self):
        for name, cfg in self.stages_config.items():
            self.stages[name] = ProjectStage(
                name=name,
                base_params=cfg['base_params'],
                risk_factors=cfg.get('risk_factors', {}),
                dependencies=cfg.get('dependencies', [])
            )

    def run_simulation(self):
        """Jalankan simulasi Monte Carlo, hitung waktu total dengan dependensi."""
        np.random.seed(42)
        results     = pd.DataFrame(index=range(self.num_simulations))
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times   = pd.DataFrame(index=range(self.num_simulations))

        for name, stage in self.stages.items():
            results[name] = stage.sample_duration(self.num_simulations)

        for name, stage in self.stages.items():
            deps = stage.dependencies
            if not deps:
                start_times[name] = 0
            else:
                start_times[name] = end_times[deps].max(axis=1)
            end_times[name] = start_times[name] + results[name]

        results['Total_Duration'] = end_times.max(axis=1)
        for name in self.stages:
            results[f'{name}_Start']  = start_times[name]
            results[f'{name}_Finish'] = end_times[name]

        self.simulation_results = results
        return results

    def critical_path_probability(self):
        """Probabilitas tiap tahapan berada di critical path."""
        total = self.simulation_results['Total_Duration']
        out = {}
        for name in self.stages:
            finish = self.simulation_results[f'{name}_Finish']
            corr   = self.simulation_results[name].corr(total)
            is_crit = (finish + 0.01) >= total
            out[name] = {
                'probability': float(np.mean(is_crit)),
                'correlation': float(corr),
                'avg_duration': float(self.simulation_results[name].mean())
            }
        return pd.DataFrame(out).T

    def risk_contribution(self):
        """Kontribusi variabilitas setiap tahapan terhadap total durasi."""
        total_var = self.simulation_results['Total_Duration'].var()
        out = {}
        for name in self.stages:
            covar = self.simulation_results[name].cov(self.simulation_results['Total_Duration'])
            out[name] = {
                'variance': float(self.simulation_results[name].var()),
                'contribution_percent': float((covar / total_var) * 100),
                'std_dev': float(self.simulation_results[name].std())
            }
        return pd.DataFrame(out).T


# ============================================================================
# 3. KONFIGURASI TAHAPAN PROYEK KONSTRUKSI FITE (satuan: bulan)
# ============================================================================
DEFAULT_CONFIG = {
    "Persiapan_Lahan": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "cuaca_buruk":      {"type": "discrete",   "probability": 0.3, "impact": 0.3},
            "perizinan_mundur": {"type": "discrete",   "probability": 0.2, "impact": 0.4},
            "produktivitas":    {"type": "continuous", "mean": 1.0,        "std": 0.15}
        }
    },
    "Fondasi_Struktur": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "cuaca_buruk":        {"type": "discrete",   "probability": 0.35, "impact": 0.25},
            "material_terlambat": {"type": "discrete",   "probability": 0.25, "impact": 0.3},
            "produktivitas":      {"type": "continuous", "mean": 1.0,         "std": 0.2}
        },
        "dependencies": ["Persiapan_Lahan"]
    },
    "Struktur_Bangunan": {
        "base_params": {"optimistic": 5, "most_likely": 7, "pessimistic": 10},
        "risk_factors": {
            "cuaca_buruk":        {"type": "discrete",   "probability": 0.4, "impact": 0.2},
            "material_terlambat": {"type": "discrete",   "probability": 0.3, "impact": 0.25},
            "perubahan_desain":   {"type": "discrete",   "probability": 0.2, "impact": 0.35},
            "produktivitas":      {"type": "continuous", "mean": 1.0,        "std": 0.2}
        },
        "dependencies": ["Fondasi_Struktur"]
    },
    "Instalasi_MEP": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "material_teknis_khusus": {"type": "discrete",   "probability": 0.35, "impact": 0.4},
            "cuaca_buruk":            {"type": "discrete",   "probability": 0.2,  "impact": 0.15},
            "produktivitas":          {"type": "continuous", "mean": 1.0,          "std": 0.2}
        },
        "dependencies": ["Struktur_Bangunan"]
    },
    "Interior_Lab_Kelas": {
        "base_params": {"optimistic": 3, "most_likely": 4, "pessimistic": 7},
        "risk_factors": {
            "perubahan_desain_lab":   {"type": "discrete",   "probability": 0.4, "impact": 0.35},
            "material_teknis_khusus": {"type": "discrete",   "probability": 0.3, "impact": 0.3},
            "produktivitas":          {"type": "continuous", "mean": 1.0,        "std": 0.25}
        },
        "dependencies": ["Instalasi_MEP"]
    },
    "Finishing_Eksterior": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "cuaca_buruk":    {"type": "discrete",   "probability": 0.45, "impact": 0.3},
            "produktivitas":  {"type": "continuous", "mean": 1.0,         "std": 0.15}
        },
        "dependencies": ["Struktur_Bangunan"]
    },
    "Instalasi_Peralatan_Lab": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 4},
        "risk_factors": {
            "material_teknis_khusus": {"type": "discrete",   "probability": 0.5, "impact": 0.5},
            "perubahan_desain_lab":   {"type": "discrete",   "probability": 0.25,"impact": 0.3},
            "produktivitas":          {"type": "continuous", "mean": 1.0,        "std": 0.2}
        },
        "dependencies": ["Interior_Lab_Kelas"]
    },
    "Uji_Coba_Serah_Terima": {
        "base_params": {"optimistic": 1, "most_likely": 1, "pessimistic": 2},
        "risk_factors": {
            "temuan_masalah": {"type": "discrete",   "probability": 0.3, "impact": 0.6},
            "produktivitas":  {"type": "continuous", "mean": 1.0,        "std": 0.1}
        },
        "dependencies": ["Finishing_Eksterior", "Instalasi_Peralatan_Lab"]
    }
}

# Deskripsi singkat tiap tahapan (untuk UI)
STAGE_DESC = {
    "Persiapan_Lahan":         "Pembersihan lahan, survei, dan perizinan",
    "Fondasi_Struktur":        "Galian, pondasi tiang pancang, sloof",
    "Struktur_Bangunan":       "Kolom, balok, pelat lantai 5 lantai",
    "Instalasi_MEP":           "Mekanikal, Elektrikal, Plumbing",
    "Interior_Lab_Kelas":      "Dinding, lantai, plafon lab & kelas",
    "Finishing_Eksterior":     "Fasad, cat eksterior, jalan akses",
    "Instalasi_Peralatan_Lab": "Peralatan VR/AR, komputer, lab elektro",
    "Uji_Coba_Serah_Terima":   "Komisioning, uji fungsi, serah terima"
}

# ============================================================================
# 4. FUNGSI VISUALISASI PLOTLY
# ============================================================================
def plot_distribution(results):
    total = results['Total_Duration']
    mean_d, med_d = total.mean(), np.median(total)
    ci80 = np.percentile(total, [10, 90])
    ci95 = np.percentile(total, [2.5, 97.5])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total, nbinsx=60, name='Distribusi Durasi',
        marker_color='royalblue', opacity=0.75, histnorm='probability density'
    ))
    fig.add_vline(x=mean_d, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_d:.1f} bln", annotation_position="top right")
    fig.add_vline(x=med_d,  line_dash="dash", line_color="green",
                  annotation_text=f"Median: {med_d:.1f} bln")
    fig.add_vrect(x0=ci80[0], x1=ci80[1], fillcolor="yellow",  opacity=0.15,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci95[0], x1=ci95[1], fillcolor="orange",  opacity=0.08,
                  annotation_text="95% CI", line_width=0)
    fig.update_layout(
        title='Distribusi Total Durasi Pembangunan Gedung FITE',
        xaxis_title='Durasi Total (Bulan)',
        yaxis_title='Densitas Probabilitas',
        height=480, showlegend=False
    )
    stats = {'mean': mean_d, 'median': med_d, 'std': total.std(),
             'min': total.min(), 'max': total.max(),
             'ci80': ci80, 'ci95': ci95}
    return fig, stats


def plot_completion_probability(results):
    deadlines = np.arange(10, 35, 0.5)
    probs = [np.mean(results['Total_Duration'] <= d) for d in deadlines]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deadlines, y=probs, mode='lines',
        name='Probabilitas Selesai',
        line=dict(color='darkblue', width=3),
        fill='tozeroy', fillcolor='rgba(100,149,237,0.2)'
    ))
    for y_val, color, label in [(0.5, 'red', '50%'), (0.8, 'green', '80%'), (0.95, 'blue', '95%')]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right")

    # Tandai deadline skenario utama (16, 20, 24 bulan)
    for dl in [16, 20, 24]:
        idx = np.argmin(np.abs(deadlines - dl))
        p   = probs[idx]
        fig.add_trace(go.Scatter(
            x=[dl], y=[p], mode='markers+text',
            marker=dict(size=14, color='crimson', symbol='diamond'),
            text=[f"{dl} bln<br>{p:.1%}"],
            textposition="top center",
            showlegend=False
        ))

    fig.add_vrect(x0=16, x1=24, fillcolor="lightgreen", opacity=0.08,
                  annotation_text="Skenario Deadline", line_width=0)
    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek Pembangunan',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[10, 35],
        height=480
    )
    return fig


def plot_critical_path(critical_df):
    df = critical_df.sort_values('probability', ascending=True)
    colors = ['#DC2626' if p > 0.7 else '#F87171' if p > 0.4 else '#FCA5A5'
              for p in df['probability']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[s.replace('_', ' ') for s in df.index],
        x=df['probability'],
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1%}" for p in df['probability']],
        textposition='auto'
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray",
                  annotation_text="50%")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange",
                  annotation_text="70% (kritis)")
    fig.update_layout(
        title='Probabilitas Setiap Tahapan Menjadi Critical Path',
        xaxis_title='Probabilitas Critical Path',
        xaxis_range=[0, 1.05],
        height=450
    )
    return fig


def plot_boxplot(results, stages):
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    for i, name in enumerate(stages.keys()):
        fig.add_trace(go.Box(
            y=results[name],
            name=name.replace('_', '<br>'),
            boxmean='sd',
            marker_color=colors[i % len(colors)],
            boxpoints='outliers'
        ))
    fig.update_layout(
        title='Distribusi Durasi per Tahapan',
        yaxis_title='Durasi (Bulan)',
        height=450, showlegend=False
    )
    return fig


def plot_risk_contribution(risk_df):
    df = risk_df.sort_values('contribution_percent', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[n.replace('_', '<br>') for n in df.index],
        y=df['contribution_percent'],
        marker_color=px.colors.qualitative.Pastel,
        text=[f"{v:.1f}%" for v in df['contribution_percent']],
        textposition='auto'
    ))
    fig.update_layout(
        title='Kontribusi Risiko Tiap Tahapan terhadap Variabilitas Total',
        yaxis_title='Kontribusi Variabilitas (%)',
        height=400
    )
    return fig


def plot_correlation_heatmap(results, stages):
    corr = results[list(stages.keys())].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[n.replace('_', '<br>') for n in corr.columns],
        y=[n.replace('_', '<br>') for n in corr.index],
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig.update_layout(
        title='Matriks Korelasi Antar Tahapan Konstruksi',
        height=500
    )
    return fig


def plot_resource_impact(scenario_results):
    labels = [f"S{i+1}: {s['stage'][:15]}<br>({s['quantity']} {s['resource_type']})"
              for i, s in enumerate(scenario_results)]
    reductions = [s['duration_reduction'] for s in scenario_results]
    rois       = [s['roi']                for s in scenario_results]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Pengurangan Durasi per Skenario (Bulan)',
                                        'ROI per Skenario (%)'))

    fig.add_trace(go.Bar(
        x=labels, y=reductions,
        marker_color='seagreen', text=[f"{r:.2f} bln" for r in reductions],
        textposition='auto', name='Pengurangan'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=rois,
        marker_color=['limegreen' if r > 0 else 'tomato' for r in rois],
        text=[f"{r:.0f}%" for r in rois],
        textposition='auto', name='ROI'
    ), row=1, col=2)
    fig.add_hline(y=0, row=1, col=2, line_color='black', line_width=1)

    fig.update_layout(height=420, showlegend=False,
                      title_text='Dampak Penambahan Resource terhadap Proyek Konstruksi FITE')
    return fig


# ============================================================================
# 5. ANALISIS RESOURCE OPTIMIZATION
# ============================================================================
RESOURCE_COSTS = {
    'tukang_ahli':     {'cost_per_month': 8_000_000,  'productivity_gain': 0.20},
    'alat_berat':      {'cost_per_month': 25_000_000, 'productivity_gain': 0.30},
    'insinyur_sipil':  {'cost_per_month': 15_000_000, 'productivity_gain': 0.25},
    'insinyur_mep':    {'cost_per_month': 15_000_000, 'productivity_gain': 0.25},
    'mandor_senior':   {'cost_per_month': 10_000_000, 'productivity_gain': 0.15},
}

def analyze_resource_scenario(results, stages, stage_name, resource_type, quantity, duration_months):
    rp = RESOURCE_COSTS[resource_type]
    improvement = 1.0 - (rp['productivity_gain'] * min(quantity / 2, 1.0))

    mod_results = results.copy()
    mod_results[stage_name] = mod_results[stage_name] * improvement

    # Hitung ulang total durasi dengan dependensi
    start_t = {}
    end_t   = {}
    for name, stage in stages.items():
        deps = stage.dependencies
        if not deps:
            start_t[name] = np.zeros(len(results))
        else:
            start_t[name] = np.maximum.reduce([end_t[d] for d in deps])
        dur = mod_results[stage_name].values if name == stage_name else results[name].values
        end_t[name] = start_t[name] + dur

    new_total = np.maximum.reduce(list(end_t.values()))
    baseline  = results['Total_Duration'].mean()
    optimized = new_total.mean()
    reduction = baseline - optimized
    pct_imp   = (reduction / baseline) * 100

    total_cost   = rp['cost_per_month'] * quantity * duration_months
    project_cost_per_month = 500_000_000  # Rp 500 juta/bulan
    saving       = reduction * project_cost_per_month
    net_benefit  = saving - total_cost
    roi          = (net_benefit / total_cost * 100) if total_cost > 0 else 0

    # Probabilitas deadline skenario
    deadlines = [16, 20, 24]
    deadline_impact = {}
    for dl in deadlines:
        base_p = np.mean(results['Total_Duration'] <= dl)
        opt_p  = np.mean(new_total <= dl)
        deadline_impact[dl] = {'baseline': base_p, 'optimized': opt_p, 'improvement': opt_p - base_p}

    return {
        'stage': stage_name,
        'resource_type': resource_type,
        'quantity': quantity,
        'duration_months': duration_months,
        'baseline_mean': baseline,
        'optimized_mean': optimized,
        'duration_reduction': reduction,
        'percent_improvement': pct_imp,
        'total_cost': total_cost,
        'net_benefit': net_benefit,
        'roi': roi,
        'deadline_impact': deadline_impact
    }


# ============================================================================
# 6. FUNGSI UTAMA STREAMLIT
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">🏗️ Simulasi Monte Carlo<br>Estimasi Waktu Pembangunan Gedung FITE</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Studi Kasus:</b> Proyek pembangunan Gedung FITE 5 lantai dengan fasilitas lengkap
    (ruang kelas, laboratorium komputer, laboratorium elektro, laboratorium mobile, laboratorium VR/AR,
    laboratorium game, ruang dosen, toilet, dan ruang serbaguna).<br>
    Simulasi ini menggunakan <b>Monte Carlo</b> untuk memodelkan ketidakpastian durasi konstruksi
    akibat cuaca buruk, keterlambatan material, perubahan desain, dan produktivitas pekerja.
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")

    num_simulations = st.sidebar.slider(
        'Jumlah Iterasi Simulasi', 2000, 50000, 20000, 1000,
        help='Semakin banyak iterasi = hasil lebih akurat, tapi lebih lama'
    )

    st.sidebar.markdown("### 📋 Parameter Tahapan Proyek")
    st.sidebar.caption("Durasi dalam **bulan**")

    config = {}
    for stage_name, default in DEFAULT_CONFIG.items():
        bp = default['base_params']
        with st.sidebar.expander(f"⚙️ {stage_name.replace('_', ' ')}", expanded=False):
            st.caption(STAGE_DESC[stage_name])
            o = st.number_input("Optimistik",  1, 24, bp['optimistic'],  key=f"o_{stage_name}")
            m = st.number_input("Most Likely", 1, 24, bp['most_likely'],  key=f"m_{stage_name}")
            p = st.number_input("Pesimistik",  1, 36, bp['pessimistic'], key=f"p_{stage_name}")

        new_cfg = {k: v for k, v in default.items()}
        new_cfg['base_params'] = {'optimistic': o, 'most_likely': m, 'pessimistic': p}
        config[stage_name] = new_cfg

    run_btn = st.sidebar.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.78rem;color:#666;">
    <b>Keterangan:</b><br>
    • Optimistik  : Estimasi terbaik<br>
    • Most Likely : Estimasi paling realistis<br>
    • Pesimistik  : Estimasi terburuk<br>
    • Semua durasi dalam <b>bulan</b>
    </div>
    """, unsafe_allow_html=True)

    # ── SESSION STATE ────────────────────────────────────────────────────
    if 'results'   not in st.session_state: st.session_state.results   = None
    if 'simulator' not in st.session_state: st.session_state.simulator = None

    if run_btn:
        with st.spinner('Menjalankan simulasi Monte Carlo... harap tunggu...'):
            sim = MonteCarloConstructionSimulation(config, num_simulations)
            res = sim.run_simulation()
            st.session_state.results   = res
            st.session_state.simulator = sim
            st.session_state.num_sim   = num_simulations
        st.success(f'✅ Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.')

    # ── TAMPILKAN HASIL ──────────────────────────────────────────────────
    if st.session_state.results is not None:
        results   = st.session_state.results
        simulator = st.session_state.simulator
        total_dur = results['Total_Duration']

        # ── USE CASE 1: Statistik Total Durasi ───────────────────────────
        st.markdown('<h2 class="sub-header">📊 UC-1 · Total Waktu Penyelesaian Proyek</h2>',
                    unsafe_allow_html=True)

        mean_d = total_dur.mean()
        med_d  = np.median(total_dur)
        ci80   = np.percentile(total_dur, [10, 90])
        ci95   = np.percentile(total_dur, [2.5, 97.5])

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Rata-rata Durasi",   f"{mean_d:.1f} bln"),
            (c2, "Median Durasi",      f"{med_d:.1f} bln"),
            (c3, "80% CI",             f"{ci80[0]:.1f} – {ci80[1]:.1f} bln"),
            (c4, "95% CI",             f"{ci95[0]:.1f} – {ci95[1]:.1f} bln"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <h3>{val}</h3><p>{label}</p>
            </div>""", unsafe_allow_html=True)

        fig_dist, stats = plot_distribution(results)
        st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("📋 Statistik Deskriptif Lengkap"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Statistik Durasi Total:**")
                st.write(f"- Rata-rata  : {stats['mean']:.2f} bulan")
                st.write(f"- Median     : {stats['median']:.2f} bulan")
                st.write(f"- Std Dev    : {stats['std']:.2f} bulan")
                st.write(f"- Minimum    : {stats['min']:.2f} bulan")
                st.write(f"- Maksimum   : {stats['max']:.2f} bulan")
            with c2:
                st.write("**Confidence Intervals:**")
                st.write(f"- 80% CI : [{ci80[0]:.1f}, {ci80[1]:.1f}] bulan")
                st.write(f"- 95% CI : [{ci95[0]:.1f}, {ci95[1]:.1f}] bulan")
                safety   = np.percentile(total_dur, 80) - mean_d
                conting  = np.percentile(total_dur, 95) - mean_d
                st.write(f"- Safety buffer (80%): **{safety:.1f} bulan**")
                st.write(f"- Contingency (95%)  : **{conting:.1f} bulan**")

        # ── USE CASE 2: Risiko Keterlambatan ─────────────────────────────
        st.markdown('<h2 class="sub-header">⚠️ UC-2 · Risiko Keterlambatan Akibat Ketidakpastian</h2>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_risk = plot_risk_contribution(simulator.risk_contribution())
            st.plotly_chart(fig_risk, use_container_width=True)
        with c2:
            fig_corr = plot_correlation_heatmap(results, simulator.stages)
            st.plotly_chart(fig_corr, use_container_width=True)

        risk_df = simulator.risk_contribution().sort_values('contribution_percent', ascending=False)
        st.markdown("**Kontribusi Risiko per Tahapan:**")
        risk_display = risk_df[['std_dev', 'variance', 'contribution_percent']].copy()
        risk_display.columns = ['Std Dev (bln)', 'Varians', 'Kontribusi (%)']
        risk_display = risk_display.round(3)
        st.dataframe(risk_display, use_container_width=True)

        with st.expander("ℹ️ Interpretasi Risiko"):
            top_risk = risk_df.index[0].replace('_', ' ')
            st.markdown(f"""
            <div class="warning-box">
            Faktor ketidakpastian yang paling mempengaruhi total durasi proyek:
            <ol>
            <li><b>Cuaca buruk</b> – menunda pekerjaan struktural dan eksterior (prob. 35–45%, dampak 20–30%)</li>
            <li><b>Keterlambatan material teknis khusus</b> – MEP dan peralatan lab (prob. 30–50%, dampak 30–50%)</li>
            <li><b>Perubahan desain laboratorium</b> – revisi spesifikasi lab VR/AR, game, elektro (prob. 20–40%, dampak 30–35%)</li>
            <li><b>Produktivitas pekerja</b> – variasi harian tenaga kerja (std dev 0.15–0.25)</li>
            </ol>
            Tahapan dengan kontribusi risiko tertinggi: <b>{top_risk}</b>
            </div>
            """, unsafe_allow_html=True)

        # ── USE CASE 3: Critical Path ─────────────────────────────────────
        st.markdown('<h2 class="sub-header">🔴 UC-3 · Critical Path – Tahapan Paling Kritis</h2>',
                    unsafe_allow_html=True)

        crit_df = simulator.critical_path_probability()
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(plot_critical_path(crit_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_boxplot(results, simulator.stages), use_container_width=True)

        top3 = crit_df.sort_values('probability', ascending=False).head(3)
        st.markdown("**3 Tahapan Paling Kritis:**")
        for i, (stage, row) in enumerate(top3.iterrows(), 1):
            color = "#DC2626" if row['probability'] > 0.7 else "#F97316"
            st.markdown(f"""
            <div style="padding:0.5rem;background:#FFF5F5;border-left:4px solid {color};margin:0.3rem 0;border-radius:5px;">
            <b>{i}. {stage.replace('_',' ')}</b> — 
            Prob. critical: <b>{row['probability']:.1%}</b> | 
            Korelasi: {row['correlation']:.2f} | 
            Rata-rata: {row['avg_duration']:.1f} bln
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Tabel Lengkap Critical Path"):
            st.dataframe(crit_df.sort_values('probability', ascending=False).round(3),
                         use_container_width=True)

        # ── USE CASE 4: Probabilitas Deadline ────────────────────────────
        st.markdown('<h2 class="sub-header">🎯 UC-4 · Probabilitas Penyelesaian Berbagai Skenario Deadline</h2>',
                    unsafe_allow_html=True)

        st.plotly_chart(plot_completion_probability(results), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        for col, deadline, label in [
            (col_a, 16, "🟡 16 Bulan (Agresif)"),
            (col_b, 20, "🟢 20 Bulan (Realistis)"),
            (col_c, 24, "🔵 24 Bulan (Konservatif)")
        ]:
            prob_on   = np.mean(total_dur <= deadline)
            prob_late = 1 - prob_on
            exp_over  = max(0, np.percentile(total_dur[total_dur > deadline], 75)
                            - deadline) if prob_late > 0 else 0
            col.metric(label=label,
                       value=f"{prob_on:.1%}",
                       delta=f"Risiko terlambat: {prob_late:.1%}",
                       delta_color="inverse")
            col.caption(f"Estimasi keterlambatan (P75): +{exp_over:.1f} bln")

        with st.expander("📅 Analisis Deadline Kustom"):
            dl_input = st.slider("Pilih deadline target (bulan):", 10, 36, 20)
            p_on_time = np.mean(total_dur <= dl_input)
            days_over = max(0, np.percentile(total_dur, 95) - dl_input)
            st.metric(f"Probabilitas selesai ≤ {dl_input} bulan",
                      value=f"{p_on_time:.1%}",
                      delta=f"Potensi keterlambatan P95: {days_over:.1f} bln",
                      delta_color="inverse")

        # ── USE CASE 5: Pengaruh Penambahan Resource ──────────────────────
        st.markdown('<h2 class="sub-header">🔧 UC-5 · Dampak Penambahan Resource terhadap Percepatan Proyek</h2>',
                    unsafe_allow_html=True)

        # Skenario default penambahan resource
        scenarios_def = [
            {'stage': 'Struktur_Bangunan',       'resource_type': 'alat_berat',    'quantity': 2, 'duration_months': 5},
            {'stage': 'Struktur_Bangunan',       'resource_type': 'tukang_ahli',   'quantity': 5, 'duration_months': 6},
            {'stage': 'Instalasi_MEP',           'resource_type': 'insinyur_mep',  'quantity': 2, 'duration_months': 3},
            {'stage': 'Interior_Lab_Kelas',      'resource_type': 'tukang_ahli',   'quantity': 4, 'duration_months': 4},
            {'stage': 'Instalasi_Peralatan_Lab', 'resource_type': 'insinyur_sipil','quantity': 1, 'duration_months': 2},
            {'stage': 'Fondasi_Struktur',        'resource_type': 'alat_berat',    'quantity': 1, 'duration_months': 2},
            {'stage': 'Finishing_Eksterior',     'resource_type': 'tukang_ahli',   'quantity': 3, 'duration_months': 2},
        ]

        with st.spinner("Menghitung dampak resource..."):
            scenario_results = [
                analyze_resource_scenario(results, simulator.stages,
                                          s['stage'], s['resource_type'],
                                          s['quantity'], s['duration_months'])
                for s in scenarios_def
            ]

        st.plotly_chart(plot_resource_impact(scenario_results), use_container_width=True)

        # Tabel ringkasan
        tbl = []
        for i, r in enumerate(scenario_results, 1):
            tbl.append({
                "No": i,
                "Tahapan": r['stage'].replace('_', ' '),
                "Resource": r['resource_type'].replace('_', ' '),
                "Jumlah": r['quantity'],
                "Durasi (bln)": r['duration_months'],
                "Pengurangan (bln)": round(r['duration_reduction'], 2),
                "Improvement (%)": round(r['percent_improvement'], 1),
                "Biaya (Rp juta)": round(r['total_cost'] / 1_000_000, 1),
                "Net Benefit (Rp juta)": round(r['net_benefit'] / 1_000_000, 1),
                "ROI (%)": round(r['roi'], 1)
            })
        df_tbl = pd.DataFrame(tbl)
        st.dataframe(df_tbl, use_container_width=True)

        # Rekomendasi terbaik
        best_roi = max(scenario_results, key=lambda x: x['roi'])
        best_red = max(scenario_results, key=lambda x: x['duration_reduction'])
        st.markdown(f"""
        <div class="info-box">
        🏆 <b>Rekomendasi Resource Optimal:</b><br>
        • <b>ROI Tertinggi</b>: Tambahkan <b>{best_roi['quantity']} {best_roi['resource_type'].replace('_', ' ')}</b>
          di tahap <i>{best_roi['stage'].replace('_', ' ')}</i>
          → ROI <b>{best_roi['roi']:.0f}%</b>,
          hemat <b>{best_roi['duration_reduction']:.1f} bulan</b><br>
        • <b>Percepatan Terbesar</b>: Tahap <i>{best_red['stage'].replace('_', ' ')}</i>
          dengan <b>{best_red['quantity']} {best_red['resource_type'].replace('_', ' ')}</b>
          → percepat <b>{best_red['duration_reduction']:.1f} bulan</b>
        </div>
        """, unsafe_allow_html=True)

        # Dampak per deadline
        with st.expander("📅 Detail Dampak Resource pada Deadline (16/20/24 Bulan)"):
            for r in scenario_results:
                st.markdown(f"**{r['stage'].replace('_',' ')} · {r['quantity']} {r['resource_type'].replace('_',' ')}**")
                dc, dc2, dc3 = st.columns(3)
                for col, dl in [(dc, 16), (dc2, 20), (dc3, 24)]:
                    imp = r['deadline_impact'][dl]
                    col.metric(f"Deadline {dl} bln",
                               value=f"{imp['optimized']:.1%}",
                               delta=f"+{imp['improvement']:.1%}")
                st.markdown("---")

        # ── RINGKASAN AKHIR ───────────────────────────────────────────────
        st.markdown('<h2 class="sub-header">📝 Ringkasan Hasil Simulasi</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        <b>Hasil Simulasi Monte Carlo – Gedung FITE 5 Lantai</b><br><br>
        <b>1. Estimasi Total Durasi:</b><br>
           &nbsp;&nbsp;• Rata-rata: <b>{mean_d:.1f} bulan</b> |
           80% CI: [{ci80[0]:.1f}, {ci80[1]:.1f}] bulan |
           95% CI: [{ci95[0]:.1f}, {ci95[1]:.1f}] bulan<br><br>
        <b>2. Risiko Keterlambatan:</b><br>
           &nbsp;&nbsp;• Faktor dominan: cuaca buruk, material teknis khusus, perubahan desain lab<br>
           &nbsp;&nbsp;• Tahap paling variabel: <b>{risk_df.index[0].replace('_',' ')}</b>
           (kontribusi {risk_df.iloc[0]['contribution_percent']:.1f}%)<br><br>
        <b>3. Critical Path:</b><br>
           &nbsp;&nbsp;• Tahapan paling kritis: <b>{crit_df.sort_values('probability',ascending=False).index[0].replace('_',' ')}</b>
           ({crit_df.sort_values('probability',ascending=False).iloc[0]['probability']:.1%})<br><br>
        <b>4. Probabilitas Deadline:</b><br>
           &nbsp;&nbsp;• 16 bulan: <b>{np.mean(total_dur <= 16):.1%}</b> |
           20 bulan: <b>{np.mean(total_dur <= 20):.1%}</b> |
           24 bulan: <b>{np.mean(total_dur <= 24):.1%}</b><br><br>
        <b>5. Rekomendasi Resource:</b><br>
           &nbsp;&nbsp;• ROI terbaik: {best_roi['quantity']} {best_roi['resource_type'].replace('_',' ')}
           di {best_roi['stage'].replace('_',' ')} → ROI {best_roi['roi']:.0f}%<br>
           &nbsp;&nbsp;• Percepatan terbesar: {best_red['duration_reduction']:.1f} bulan
           dari {best_red['stage'].replace('_',' ')}
        </div>
        """, unsafe_allow_html=True)

    else:
        # Belum ada simulasi
        st.markdown("""
        <div style="text-align:center;padding:4rem;background:#F8FAFC;border-radius:12px;">
            <h3>🏗️ Siap memulai simulasi?</h3>
            <p>Atur parameter tahapan konstruksi di <b>sidebar kiri</b>,
               lalu klik <b>"🚀 Jalankan Simulasi"</b>.</p>
            <p>Hasil analisis 5 use case akan ditampilkan di sini.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="sub-header">📋 Preview Konfigurasi Tahapan</h2>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (name, cfg) in enumerate(DEFAULT_CONFIG.items()):
            bp = cfg['base_params']
            cols[i % 2].markdown(f"""
            <div class="stage-card">
            <b>{name.replace('_',' ')}</b><br>
            {STAGE_DESC[name]}<br>
            O: {bp['optimistic']} bln · ML: {bp['most_likely']} bln · P: {bp['pessimistic']} bln
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;font-size:0.85rem;">
    <b>Simulasi Monte Carlo – Estimasi Waktu Pembangunan Gedung FITE</b> |
    Modul Praktikum 5 – Pemodelan dan Simulasi 2026<br>
    ⚠️ Hasil ini merupakan estimasi probabilistik, bukan prediksi pasti.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
