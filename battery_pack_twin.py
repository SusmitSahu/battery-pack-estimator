"""
╔══════════════════════════════════════════════════════════════════════╗
║   Battery Pack Digital Twin — Streamlit + PyBaMM                    ║
║   Architecture: Dynamic Inputs → Pack Core (PyBaMM) →               ║
║                 SoC / SoH / RuL Outputs → Feedback Loop             ║
║                                                                      ║
║   Install:  pip install pybamm streamlit plotly pandas scipy numpy   ║
║   Run:      streamlit run battery_pack_twin.py                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pybamm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.interpolate as interp
import io
import json
import time
from copy import deepcopy

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery Pack Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main .block-container{padding-top:0.8rem;padding-bottom:0.5rem}
  h1{font-size:1.35rem!important;font-weight:700}
  h2{font-size:1.05rem!important;font-weight:600}
  h3{font-size:0.9rem!important;font-weight:600}
  .metric-card{
    background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    border-radius:10px;padding:14px 18px;border:1px solid #334155;
    margin-bottom:6px;color:#fff
  }
  .metric-val{font-size:1.6rem;font-weight:700;color:#34d399}
  .metric-label{font-size:0.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em}
  .metric-sub{font-size:0.78rem;color:#cbd5e1;margin-top:2px}
  .section-banner{
    background:#1e293b;border-left:3px solid #34d399;
    padding:6px 12px;border-radius:4px;margin:8px 0 4px 0;
    font-size:0.82rem;color:#e2e8f0;font-weight:500
  }
  .flow-box{
    background:#0f172a;border:1px solid #334155;border-radius:8px;
    padding:8px 12px;font-size:0.78rem;color:#94a3b8;margin-bottom:4px
  }
  .stTabs [data-baseweb="tab"]{font-size:0.82rem}
  .stAlert{font-size:0.82rem}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PARAM_SETS   = ["Chen2020", "Marquis2019", "Ecker2015", "Ai2020"]
MODEL_OPTIONS = {
    "SPM  (fast, simple)":             "SPM",
    "SPMe (SPM + electrolyte)":        "SPMe",
    "DFN  (full physics, slower)":     "DFN",
}
SEI_OPTIONS = {
    "None":                     None,
    "Constant (fast)":          "constant",
    "EC reaction limited":      "ec reaction limited",
    "Reaction limited":         "reaction limited",
}
THERMAL_OPTIONS = {
    "Isothermal":               "isothermal",
    "Lumped (single node)":     "lumped",
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "config_snapshot" not in st.session_state:
    st.session_state.config_snapshot = {}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — BLACK BOX STATIC CONFIG
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔋 Pack Digital Twin")
    st.markdown('<div class="flow-box">⬛ Static Config (Black-Box)<br>Materials · Geometry · Pack topology · Degradation</div>', unsafe_allow_html=True)

    # ── Pack topology
    with st.expander("🗂️ Pack topology  (Ns × Np)", expanded=True):
        Ns = st.number_input("Cells in Series (Ns)", 1, 20, 2, 1,
            help="Number of cells connected in series")
        Np = st.number_input("Cells in Parallel (Np)", 1, 10, 2, 1,
            help="Number of cells connected in parallel")
        st.caption(f"Pack = **{Ns}S{Np}P** → {Ns*Np} cells total")
        cell_capacity = st.number_input("Nominal cell capacity (Ah)", 0.5, 20.0, 5.0, 0.5)
        cell_voltage_nom = st.number_input("Nominal cell voltage (V)", 2.5, 4.5, 3.6, 0.1)
        st.caption(f"Pack voltage ≈ {Ns*cell_voltage_nom:.1f} V | Pack capacity ≈ {Np*cell_capacity:.1f} Ah")

    # ── Model & chemistry
    with st.expander("⚗️ Model & chemistry", expanded=True):
        model_label = st.selectbox("Electrochemical model", list(MODEL_OPTIONS.keys()))
        param_set   = st.selectbox("Parameter set (chemistry)", PARAM_SETS)
        thermal_label = st.selectbox("Thermal model", list(THERMAL_OPTIONS.keys()))
        sei_label   = st.selectbox("SEI degradation", list(SEI_OPTIONS.keys()))
        use_plating = st.toggle("Lithium plating", value=False,
            help="Enable irreversible lithium plating (DFN/SPM only with Chen2020)")

    # ── Geometry & material overrides
    with st.expander("📐 Geometry & material overrides"):
        st.caption("Leave at 0 to use parameter-set defaults.")
        override_h = st.number_input("Electrode height (m)", 0.0, 1.0, 0.0, 0.001,
            format="%.4f", help="0 = use default from chemistry")
        override_w = st.number_input("Electrode width (m)", 0.0, 1.0, 0.0, 0.001, format="%.4f")
        override_htc = st.number_input(
            "Heat transfer coeff. H (W/m²K)", 0.0, 500.0, 0.0, 1.0,
            help="Convective HTC per cell. 0 = default. Applied to all cells.")
        override_R = st.number_input(
            "Internal resistance offset (Ω)", 0.0, 0.05, 0.0, 0.001, format="%.4f",
            help="Additional contact/pack resistance per cell")

    # ── Cell-to-cell variation
    with st.expander("📊 Cell-to-cell variation"):
        variation_pct = st.slider(
            "Capacity spread σ (%)", 0.0, 10.0, 2.0, 0.5,
            help="Gaussian spread of nominal capacity across cells. 0 = identical cells.")
        soc_spread = st.slider(
            "Initial SoC spread σ (%)", 0.0, 5.0, 1.0, 0.5,
            help="Spread of initial state of charge across cells")

    # ── Degradation parameters
    with st.expander("🔬 Degradation parameters"):
        sei_growth = st.select_slider(
            "SEI growth rate factor", [0.5, 0.75, 1.0, 1.5, 2.0, 3.0], value=1.0,
            help="Multiplier on SEI reaction rate constant")
        cycle_life_target = st.number_input(
            "Design cycle life (cycles)", 100, 2000, 500, 50,
            help="Used for RuL normalisation")

    # ── Initial conditions
    with st.expander("🔌 Initial conditions"):
        init_soc = st.slider("Initial SoC (%)", 10, 100, 100, 5,
            help="Mean initial state of charge for all cells")

    st.divider()
    st.markdown('<div class="flow-box">⬜ Dynamic Inputs<br>I · V limits · T_amb · H</div>',
                unsafe_allow_html=True)

    # ── Dynamic inputs
    with st.expander("⚡ Dynamic inputs (I, V, T_amb, H)", expanded=True):
        protocol_type = st.radio(
            "Protocol", ["Cycling (CC-CV)", "Constant discharge", "GITT", "Custom sequence"],
            horizontal=False)

        T_amb = st.slider("Ambient temperature T_amb (°C)", -20, 60, 25, 1)
        H_conv = st.number_input(
            "Convective HTC H (W/m²K)", 0.0, 200.0,
            override_htc if override_htc > 0 else 10.0, 1.0,
            help="Per-cell convective heat transfer coefficient")

        if protocol_type == "Cycling (CC-CV)":
            n_cycles     = st.slider("Cycles to simulate", 1, 20, 3)
            I_charge     = st.number_input("Charge C-rate", 0.1, 5.0, 1.0, 0.1)
            I_discharge  = st.number_input("Discharge C-rate", 0.1, 5.0, 1.0, 0.1)
            V_max        = st.number_input("Upper cut-off V (V/cell)", 3.8, 4.4, 4.2, 0.05)
            V_min        = st.number_input("Lower cut-off V (V/cell)", 2.0, 3.5, 2.5, 0.05)
            rest_min     = st.number_input("Rest time (min)", 0, 60, 10, 5)
            experiment_steps = []
            for _ in range(n_cycles):
                experiment_steps += [
                    f"Charge at {I_charge}C until {V_max:.2f}V",
                    f"Hold at {V_max:.2f}V until C/50",
                    f"Rest for {rest_min} minutes",
                    f"Discharge at {I_discharge}C until {V_min:.2f}V",
                    f"Rest for {rest_min} minutes",
                ]

        elif protocol_type == "Constant discharge":
            I_disch_cd  = st.number_input("Discharge C-rate", 0.1, 5.0, 1.0, 0.1)
            V_cutoff    = st.number_input("Cut-off voltage (V/cell)", 2.0, 3.5, 2.5, 0.05)
            experiment_steps = [f"Discharge at {I_disch_cd}C until {V_cutoff:.2f}V"]

        elif protocol_type == "GITT":
            I_pulse   = st.number_input("Pulse current (A)", 0.1, 10.0, 1.0, 0.1)
            t_pulse   = st.number_input("Pulse duration (s)", 10, 600, 60, 10)
            t_relax   = st.number_input("Relaxation time (s)", 10, 3600, 300, 10)
            n_pulses  = st.slider("Number of pulses", 1, 20, 5)
            experiment_steps = []
            for _ in range(n_pulses):
                experiment_steps += [
                    f"Discharge at {I_pulse} A for {t_pulse} seconds",
                    f"Rest for {t_relax} seconds",
                ]
        else:  # Custom
            st.caption("Enter one experiment step per line, PyBaMM syntax:")
            custom_text = st.text_area(
                "Steps",
                value=(
                    "Charge at 1C until 4.2V\n"
                    "Hold at 4.2V until C/50\n"
                    "Rest for 10 minutes\n"
                    "Discharge at 1C until 2.5V"
                ),
                height=120,
            )
            experiment_steps = [s.strip() for s in custom_text.splitlines() if s.strip()]

    # ── Solver
    with st.expander("🔧 Solver settings"):
        solver_choice = st.selectbox("Solver", ["CasadiSolver", "ScipySolver"])
        atol = st.select_slider("Absolute tolerance",
            [1e-7, 1e-6, 1e-5, 1e-4], value=1e-6)
        rtol = st.select_slider("Relative tolerance",
            [1e-7, 1e-6, 1e-5, 1e-4], value=1e-5)

    st.divider()
    run_btn = st.button("▶  Run Pack Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_key, thermal, sei, plating):
    options = {"thermal": thermal}
    if sei:
        options["SEI"] = sei
    if plating:
        options["lithium plating"] = "irreversible"
    cls_map = {
        "SPM":  pybamm.lithium_ion.SPM,
        "SPMe": pybamm.lithium_ion.SPMe,
        "DFN":  pybamm.lithium_ion.DFN,
    }
    return cls_map[model_key](options=options)


def build_param(param_set, T_amb_K, H_conv, override_h, override_w, override_R,
                cap_scale, init_soc_frac, sei_growth):
    param = pybamm.ParameterValues(param_set)
    param["Initial temperature [K]"] = T_amb_K
    param["Ambient temperature [K]"] = T_amb_K
    param["Total heat transfer coefficient [W.m-2.K-1]"] = H_conv
    if override_h > 0:
        param["Electrode height [m]"] = override_h
    if override_w > 0:
        param["Electrode width [m]"] = override_w
    # Scale capacity via electrode width (proportional scaling)
    if cap_scale != 1.0:
        try:
            w = param["Electrode width [m]"]
            param["Electrode width [m]"] = w * cap_scale
        except Exception:
            pass
    # SEI growth rate override
    if sei_growth != 1.0:
        try:
            k = param["EC reaction rate constant [m.s-1]"]
            param["EC reaction rate constant [m.s-1]"] = k * sei_growth
        except Exception:
            pass
    # Contact resistance
    if override_R > 0:
        try:
            param["Negative current collector conductivity [S.m-1]"] = \
                1.0 / (override_R * 1e-4)
        except Exception:
            pass
    # Set initial SOC
    try:
        param["Initial concentration in negative electrode [mol.m-3]"] = \
            param["Initial concentration in negative electrode [mol.m-3]"] * (init_soc_frac / 1.0)
    except Exception:
        pass
    return param


def flat1d_arr(arr):
    """
    Global helper: force any PyBaMM array-like to a clean 1-D float64.
    For (n_space, n_time) arrays, averages over the spatial axis so the
    returned length equals n_time (the time dimension).
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        r, c = a.shape
        if r == 1:
            return a[0]
        if c == 1:
            return a[:, 0]
        # (n_space, n_time) — collapse spatial axis
        return a.mean(axis=0)
    return a.reshape(-1, a.shape[-1]).mean(axis=0)


def estimate_soc(sol):
    """Estimate SoC from discharge capacity / nominal capacity."""
    try:
        Q_disch = flat1d_arr(sol["Discharge capacity [A.h]"].entries)
        Q_max   = float(np.nanmax(np.abs(Q_disch)))
        if Q_max < 1e-9:
            return 100.0
        soc_now = max(0.0, min(100.0, (1.0 - float(Q_disch[-1]) / Q_max) * 100.0))
        return float(soc_now)
    except Exception:
        return float("nan")


def estimate_soh(sol, nominal_cap):
    """SoH via discharge capacity vs nominal."""
    try:
        Q = flat1d_arr(sol["Discharge capacity [A.h]"].entries)
        actual = float(np.nanmax(np.abs(Q)))
        return min(100.0, actual / nominal_cap * 100.0)
    except Exception:
        return float("nan")


def estimate_rul(soh_pct, cycle_life):
    """RuL in cycles assuming linear fade from 100% SoH at 80% EoL."""
    eol = 80.0
    if soh_pct <= eol:
        return 0.0
    fade_rate = (100.0 - eol) / cycle_life
    if fade_rate <= 0:
        return float(cycle_life)
    return max(0.0, (soh_pct - eol) / fade_rate)


def get_sei_thickness(sol):
    try:
        arr = flat1d_arr(sol["Negative SEI thickness [m]"].entries)
        return float(arr[-1])
    except Exception:
        return float("nan")


def run_single_cell(cell_idx, model_key, param_set, thermal, sei, plating,
                    T_amb_K, H_conv, override_h, override_w, override_R,
                    cap_scale, init_soc_frac, sei_growth, experiment_steps,
                    solver_choice, atol, rtol, nominal_cap):
    """Simulate one cell and return a result dict."""
    try:
        model  = build_model(model_key, thermal, sei, plating)
        param  = build_param(param_set, T_amb_K, H_conv, override_h, override_w,
                             override_R, cap_scale, init_soc_frac, sei_growth)
        exp    = pybamm.Experiment(experiment_steps)
        solver = (pybamm.CasadiSolver(atol=atol, rtol=rtol)
                  if solver_choice == "CasadiSolver"
                  else pybamm.ScipySolver(atol=atol, rtol=rtol))
        sim    = pybamm.Simulation(model, parameter_values=param,
                                   experiment=exp, solver=solver)
        sol    = sim.solve()

        def to_1d(arr, n_time=None):
            """
            Coerce any PyBaMM output to a 1-D time-series of length n_time.

            PyBaMM variables can come back as:
              • 1-D (n_time,)                  → use directly
              • 2-D (1, n_time) or (n_time, 1) → squeeze
              • 2-D (n_space, n_time)           → mean over spatial axis 0
                  e.g. Cell temperature [C] with lumped model → (60, 145)
            """
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                return a
            if a.ndim == 2:
                r, c = a.shape
                if r == 1:          # (1, T)  — single spatial node
                    return a[0]
                if c == 1:          # (T, 1)
                    return a[:, 0]
                # True 2-D: (n_space, n_time).
                # Use n_time hint when available to tell which axis is time.
                if n_time is not None and c == n_time:
                    return a.mean(axis=0)   # average over spatial nodes
                if n_time is not None and r == n_time:
                    return a.mean(axis=1)
                # Fallback: assume last axis is time
                return a.mean(axis=0)
            # ndim > 2: collapse everything except the last axis
            return a.reshape(-1, a.shape[-1]).mean(axis=0)

        # Time is always 1-D from PyBaMM
        t = np.asarray(sol["Time [s]"].entries, dtype=float).ravel()
        n_t = len(t)

        V      = to_1d(sol["Voltage [V]"].entries,              n_t)
        I      = to_1d(sol["Current [A]"].entries,              n_t)
        Q      = to_1d(sol["Discharge capacity [A.h]"].entries, n_t)
        T_cell = to_1d(sol["Cell temperature [C]"].entries,     n_t)

        # Safety: trim/pad everything to n_t so arrays are always aligned
        def align(a, n):
            if len(a) >= n:
                return a[:n]
            return np.pad(a, (0, n - len(a)), constant_values=a[-1])

        V      = align(V,      n_t)
        I      = align(I,      n_t)
        Q      = align(Q,      n_t)
        T_cell = align(T_cell, n_t)

        soc = estimate_soc(sol)
        soh = estimate_soh(sol, nominal_cap)
        sei_t = get_sei_thickness(sol)

        return {
            "cell_idx": cell_idx,
            "success":  True,
            "t": t, "V": V, "I": I, "Q": Q, "T_cell": T_cell,
            "SoC": soc, "SoH": soh, "SEI_m": sei_t,
        }
    except Exception as e:
        return {"cell_idx": cell_idx, "success": False, "error": str(e)}


def pack_aggregate(cell_results, Ns, Np):
    """
    Aggregate cell results for series-parallel pack.
    Series cells → voltages ADD, capacity stays same.
    Parallel cells → voltages same, capacities ADD.
    Simple block topology: Np parallel strings of Ns series cells.
    """
    n_total = Ns * Np
    good = [r for r in cell_results if r["success"]]
    if not good:
        return None

    # Align time grids to shortest
    t_min_len = min(len(r["t"]) for r in good)
    t_ref = good[0]["t"][:t_min_len]

    def flat1d(arr, n_time=None):
        """
        Coerce to 1-D time series. Mirrors the to_1d() logic in run_single_cell.
        (n_space, n_time) arrays are averaged over the spatial axis.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            return a
        if a.ndim == 2:
            r, c = a.shape
            if r == 1:
                return a[0]
            if c == 1:
                return a[:, 0]
            if n_time is not None and c == n_time:
                return a.mean(axis=0)
            if n_time is not None and r == n_time:
                return a.mean(axis=1)
            return a.mean(axis=0)
        return a.reshape(-1, a.shape[-1]).mean(axis=0)

    def interp_to(arr_t, arr_v, t_new):
        """Interpolate arr_v (on arr_t) onto t_new; robust to 2-D arrays and length mismatches."""
        n_t_hint = len(flat1d(arr_t))
        t_1d = flat1d(arr_t)
        v_1d = flat1d(arr_v, n_time=n_t_hint)
        t_q  = flat1d(t_new)
        # Trim to matching length if still mismatched
        n = min(len(t_1d), len(v_1d))
        t_1d, v_1d = t_1d[:n], v_1d[:n]
        # Remove duplicate time points (PyBaMM can produce them at step boundaries)
        _, uniq = np.unique(t_1d, return_index=True)
        t_1d = t_1d[uniq]
        v_1d = v_1d[uniq]
        # fill_value must be scalars, not arrays
        f = interp.interp1d(
            t_1d, v_1d,
            bounds_error=False,
            fill_value=(float(v_1d[0]), float(v_1d[-1])),
        )
        return f(t_q)

    V_strings = []  # one per series-string (Np strings)
    I_strings = []
    T_strings = []

    for p in range(Np):
        V_series = np.zeros(t_min_len)
        I_string = np.zeros(t_min_len)
        T_str    = np.zeros(t_min_len)
        for s in range(Ns):
            idx = p * Ns + s
            r = good[idx % len(good)]
            vi = interp_to(r["t"], r["V"], t_ref)
            ii = interp_to(r["t"], r["I"], t_ref)
            ti = interp_to(r["t"], r["T_cell"], t_ref)
            V_series += vi
            I_string  = ii  # same current through series
            T_str    += ti / Ns
        V_strings.append(V_series)
        I_strings.append(I_string)
        T_strings.append(T_str)

    # Parallel combination: average voltage, sum current
    V_pack = np.mean(V_strings, axis=0)
    I_pack = np.sum(I_strings, axis=0)
    T_pack = np.mean(T_strings, axis=0)
    Q_pack = interp_to(good[0]["t"], good[0]["Q"], t_ref) * Np

    # Pack-level metrics
    soc_vals = [r["SoC"] for r in good if not np.isnan(r["SoC"])]
    soh_vals = [r["SoH"] for r in good if not np.isnan(r["SoH"])]
    sei_vals = [r["SEI_m"] for r in good if not np.isnan(r.get("SEI_m", float("nan")))]

    pack_soc = float(np.mean(soc_vals)) if soc_vals else float("nan")
    pack_soh = float(np.mean(soh_vals)) if soh_vals else float("nan")
    pack_T_max = float(np.max([np.asarray(r["T_cell"], dtype=float).ravel().max() for r in good]))
    pack_T_min = float(np.min([np.asarray(r["T_cell"], dtype=float).ravel().min() for r in good]))

    return {
        "t": t_ref,
        "V_pack": V_pack, "I_pack": I_pack,
        "T_pack": T_pack, "Q_pack": Q_pack,
        "pack_soc": pack_soc, "pack_soh": pack_soh,
        "pack_T_max": pack_T_max, "pack_T_min": pack_T_min,
        "cell_results": good,
        "Ns": Ns, "Np": Np,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_flow = st.columns([3, 1])
with col_title:
    st.title("🔋 Battery Pack Digital Twin")
    st.caption(
        "Dynamic inputs (I · V · T_amb · H) → Liionpack-style PyBaMM pack core "
        "→ Estimation outputs (SoC · SoH · RuL · ΔT) → Optimisation feedback loop"
    )
with col_flow:
    st.markdown("""
    <div class="flow-box" style="font-size:0.72rem">
    ⬜ <b>Dynamic Inputs</b> I, V, T_amb, H<br>
    ⬇<br>
    ⬛ <b>Pack Core</b> (PyBaMM Solvers)<br>
    ⬇<br>
    🟩 <b>Outputs</b> SoC · SoH · RuL · ΔT<br>
    ⬇<br>
    🔄 <b>Feedback / Optimisation</b>
    </div>""", unsafe_allow_html=True)

if not run_btn and st.session_state.results is None:
    st.info(
        "**Configure** your battery pack in the sidebar (static black-box config + dynamic inputs), "
        "then click **▶ Run Pack Simulation**.\n\n"
        "The toolbox will simulate each cell in the **pack array** (with cell-to-cell variation), "
        "aggregate to pack-level voltages/temperatures, and compute **SoC, SoH, RuL** for every cell."
    )
    with st.expander("📖 Quick-start & architecture"):
        st.markdown("""
**Architecture follows the flow diagram:**

| Block | What it does |
|---|---|
| **Static Config** (sidebar) | Pack topology (NsNp), chemistry, geometry, degradation params — fixed per simulation run |
| **Dynamic Inputs** (sidebar) | Current I, voltage limits V, ambient temp T_amb, convective HTC H — change per experiment |
| **Pack Core** | Each cell simulated with PyBaMM (SPM/SPMe/DFN + thermal + SEI). Cell results aggregated for series-parallel pack. |
| **Estimation Outputs** | SoC (per cell + pack), SoH (capacity fade), RuL (cycles remaining), ΔT (thermal gradient) |
| **Feedback Loop** | Results feed back to parameter update or next-cycle configuration |

**Install & run:**
```bash
pip install pybamm streamlit plotly pandas scipy
streamlit run battery_pack_twin.py
```

**Tips:**
- Start with **SPM** model for speed. Use **DFN** for accuracy.
- **Chen2020** is most stable with SEI degradation.
- Enable **cell-to-cell variation** to see realistic pack imbalance.
- Large packs (e.g. 4S4P = 16 cells) take ~1–3 min with DFN; use SPM for fast runs.
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    n_total   = Ns * Np
    model_key = MODEL_OPTIONS[model_label]
    thermal   = THERMAL_OPTIONS[thermal_label]
    sei_opt   = SEI_OPTIONS[sei_label]

    # Generate per-cell variation seeds
    rng = np.random.default_rng(42)
    cap_scales   = np.clip(rng.normal(1.0, variation_pct / 100.0, n_total), 0.7, 1.3)
    soc_offsets  = np.clip(rng.normal(0.0, soc_spread / 100.0, n_total), -0.1, 0.1)

    T_amb_K = T_amb + 273.15
    nom_cap = cell_capacity

    st.markdown(f"### Simulating **{Ns}S{Np}P** pack ({n_total} cells) with **{model_key}** + {thermal} thermal…")
    prog = st.progress(0, text="Starting…")
    results_list = []
    errors = []
    t_start = time.time()

    for i in range(n_total):
        prog.progress((i) / n_total, text=f"Cell {i+1}/{n_total}…")
        soc_frac = min(1.0, max(0.01, (init_soc / 100.0) + soc_offsets[i]))
        r = run_single_cell(
            cell_idx=i,
            model_key=model_key,
            param_set=param_set,
            thermal=thermal,
            sei=sei_opt,
            plating=use_plating,
            T_amb_K=T_amb_K,
            H_conv=H_conv,
            override_h=override_h,
            override_w=override_w,
            override_R=override_R,
            cap_scale=float(cap_scales[i]),
            init_soc_frac=soc_frac,
            sei_growth=sei_growth,
            experiment_steps=experiment_steps,
            solver_choice=solver_choice,
            atol=atol,
            rtol=rtol,
            nominal_cap=nom_cap * float(cap_scales[i]),
        )
        results_list.append(r)
        if not r["success"]:
            errors.append(f"Cell {i+1}: {r['error']}")

    prog.progress(1.0, text="Aggregating pack results…")
    pack = pack_aggregate(results_list, Ns, Np)
    elapsed = time.time() - t_start

    # Compute RuL per cell
    for r in results_list:
        if r["success"]:
            r["RuL"] = estimate_rul(r["SoH"], cycle_life_target)
        else:
            r["RuL"] = float("nan")

    pack_rul = np.mean([r["RuL"] for r in results_list if r["success"] and not np.isnan(r["RuL"])])

    st.session_state.results = {
        "pack": pack,
        "cell_results": results_list,
        "errors": errors,
        "elapsed": elapsed,
        "Ns": Ns, "Np": Np,
        "n_total": n_total,
        "pack_rul": pack_rul,
        "cycle_life_target": cycle_life_target,
        "nominal_cap": nom_cap,
        "T_amb": T_amb,
    }
    prog.empty()
    if errors:
        with st.expander(f"⚠️ {len(errors)} cell(s) failed"):
            for e in errors:
                st.warning(e)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
res = st.session_state.results
if res is None:
    st.stop()

pack = res["pack"]
cells = res["cell_results"]
good  = [c for c in cells if c["success"]]
Ns_r, Np_r = res["Ns"], res["Np"]

if pack is None:
    st.error("All cell simulations failed. Check model/chemistry compatibility.")
    st.stop()

st.success(f"✅ {len(good)}/{res['n_total']} cells simulated successfully in {res['elapsed']:.1f}s")

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    ("Pack SoC",       f"{pack['pack_soc']:.1f}%",     "State of charge"),
    ("Pack SoH",       f"{pack['pack_soh']:.1f}%",     "Capacity retention"),
    ("RuL",            f"{res['pack_rul']:.0f} cyc",   "Remaining useful life"),
    ("Peak ΔT",        f"{pack['pack_T_max']-res['T_amb']:.1f}°C", "Max cell temp rise"),
    ("Pack V (end)",   f"{pack['V_pack'][-1]:.2f} V",  "Terminal voltage"),
    ("Pack Q",         f"{abs(pack['Q_pack'][-1]):.2f} Ah", "Discharged capacity"),
]
for col, (label, val, sub) in zip([c1,c2,c3,c4,c5,c6], kpis):
    with col:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-val">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Pack dynamics",
    "🌡️ Thermal map",
    "🔬 Per-cell SoC/SoH",
    "🗺️ 2D pack grid",
    "📊 Degradation",
    "📋 Data & Export",
])

# ── Tab 1: Pack voltage, current, temperature ─────────────────────────────────
with tab1:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Pack terminal voltage (V)", "Pack current (A)", "Mean cell temperature (°C)"),
    )
    t_h = pack["t"] / 3600
    fig.add_trace(go.Scatter(x=t_h, y=pack["V_pack"],
        name="V_pack", line=dict(color="#34d399", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=pack["I_pack"],
        name="I_pack", line=dict(color="#60a5fa", width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=pack["T_pack"],
        name="T_mean", line=dict(color="#f97316", width=1.8)), row=3, col=1)

    # Add cell envelopes (min/max V)
    if len(good) > 1:
        from scipy.interpolate import interp1d
        t_ref = pack["t"]
        V_all = []
        for r in good:
            t1 = np.asarray(r["t"], dtype=float).ravel()
            v1 = np.asarray(r["V"], dtype=float).ravel()
            _, uniq = np.unique(t1, return_index=True)
            t1, v1 = t1[uniq], v1[uniq]
            f = interp1d(t1, v1, bounds_error=False,
                         fill_value=(float(v1[0]), float(v1[-1])))
            V_all.append(f(t_ref))
        V_arr = np.array(V_all)  # (n_cells, t)
        # For pack series strings: sum Ns cells per string
        # Show envelope as series-string voltages
        fig.add_trace(go.Scatter(
            x=t_h, y=V_arr.max(axis=0) * Ns_r,
            name="V_max string", line=dict(color="#34d399", width=0.8, dash="dot"),
            opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t_h, y=V_arr.min(axis=0) * Ns_r,
            name="V_min string", line=dict(color="#f87171", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(52,211,153,0.06)", opacity=0.5), row=1, col=1)

    fig.update_xaxes(title_text="Time (h)", row=3, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="A", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_layout(height=520, template="plotly_dark", margin=dict(t=40, b=20),
                      showlegend=True, legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Thermal map ─────────────────────────────────────────────────────────
with tab2:
    fig2 = make_subplots(rows=1, cols=2,
        subplot_titles=("Cell temperature traces", "Temperature distribution (end of sim)"))
    colors_t = px.colors.sequential.Plasma

    for idx, r in enumerate(good):
        col_t = colors_t[int(idx / max(1, len(good)-1) * (len(colors_t)-1))]
        t_plot = np.asarray(r["t"], dtype=float).ravel()
        T_plot = np.asarray(r["T_cell"], dtype=float).ravel()
        fig2.add_trace(go.Scatter(
            x=t_plot / 3600, y=T_plot,
            name=f"Cell {r['cell_idx']+1}",
            line=dict(color=col_t, width=1.2), opacity=0.85,
            legendgroup="cells", showlegend=(idx < 12),
        ), row=1, col=1)

    T_ends = [float(np.asarray(r["T_cell"], dtype=float).ravel()[-1]) for r in good]
    fig2.add_trace(go.Histogram(
        x=T_ends, nbinsx=min(20, len(good)+1),
        marker_color="#f97316", name="T distribution",
    ), row=1, col=2)
    fig2.add_vline(x=np.mean(T_ends), line_dash="dash",
                   line_color="#fbbf24", annotation_text=f"Mean={np.mean(T_ends):.1f}°C",
                   row=1, col=2)

    fig2.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig2.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig2.update_yaxes(title_text="°C", row=1, col=1)
    fig2.update_yaxes(title_text="Count", row=1, col=2)
    fig2.update_layout(height=400, template="plotly_dark",
                       margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    # ΔT gradient bar
    T_arr = np.array(T_ends)
    st.metric("Thermal gradient ΔT across pack",
              f"{T_arr.max()-T_arr.min():.2f}°C",
              help="Max − min cell temperature at end of simulation")

# ── Tab 3: Per-cell SoC / SoH / RuL ──────────────────────────────────────────
with tab3:
    cell_df = pd.DataFrame([{
        "Cell #":    r["cell_idx"] + 1,
        "Row (S)":   r["cell_idx"] // Np_r + 1,
        "Col (P)":   r["cell_idx"] % Np_r + 1,
        "SoC (%)":   round(r["SoC"], 2) if r["success"] else None,
        "SoH (%)":   round(r["SoH"], 2) if r["success"] else None,
        "RuL (cyc)": round(r["RuL"], 0) if r["success"] else None,
        "T_max (°C)": round(float(np.asarray(r["T_cell"], dtype=float).ravel().max()), 2) if r["success"] else None,
        "SEI (nm)":   round(r.get("SEI_m", float("nan")) * 1e9, 3) if r["success"] else None,
        "Status":    "✅" if r["success"] else "❌",
    } for r in cells])

    st.dataframe(cell_df, use_container_width=True, height=320)

    fig3 = make_subplots(rows=1, cols=3,
        subplot_titles=("SoC per cell", "SoH per cell", "RuL per cell"))
    ok = cell_df[cell_df["Status"] == "✅"]
    fig3.add_trace(go.Bar(x=ok["Cell #"], y=ok["SoC (%)"],
        marker_color="#34d399", name="SoC"), row=1, col=1)
    fig3.add_trace(go.Bar(x=ok["Cell #"], y=ok["SoH (%)"],
        marker_color="#60a5fa", name="SoH"), row=1, col=2)
    fig3.add_trace(go.Bar(x=ok["Cell #"], y=ok["RuL (cyc)"],
        marker_color="#a78bfa", name="RuL"), row=1, col=3)
    for col_i, yref, color in [(1,"SoC (%)", "#34d399"),
                                (2,"SoH (%)", "#60a5fa")]:
        if not ok.empty:
            mean_v = ok[yref].mean()
            fig3.add_hline(y=mean_v, line_dash="dash", line_color=color,
                           annotation_text=f"mean {mean_v:.1f}", row=1, col=col_i)
    fig3.update_layout(height=320, template="plotly_dark",
                       margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: 2D pack grid heatmap ───────────────────────────────────────────────
with tab4:
    st.markdown("### 2D Pack Grid — Spatial view of cell health")

    metric_choice = st.selectbox(
        "Metric to display on grid",
        ["SoC (%)", "SoH (%)", "RuL (cyc)", "T_max (°C)", "SEI (nm)"],
    )

    grid_vals = np.full((Ns_r, Np_r), np.nan)
    for r in good:
        row_i = r["cell_idx"] // Np_r
        col_i = r["cell_idx"] % Np_r
        val_map = {
            "SoC (%)":   r["SoC"],
            "SoH (%)":   r["SoH"],
            "RuL (cyc)": r["RuL"],
            "T_max (°C)": float(np.asarray(r["T_cell"], dtype=float).ravel().max()),
            "SEI (nm)":  r.get("SEI_m", float("nan")) * 1e9,
        }
        grid_vals[row_i, col_i] = val_map.get(metric_choice, np.nan)

    cmap = {
        "SoC (%)":    "Greens",
        "SoH (%)":    "Blues",
        "RuL (cyc)":  "Purples",
        "T_max (°C)": "Reds",
        "SEI (nm)":   "Oranges",
    }.get(metric_choice, "Viridis")

    y_labels = [f"Series {i+1}" for i in range(Ns_r)]
    x_labels = [f"Parallel {j+1}" for j in range(Np_r)]

    # Custom annotations: value + cell number
    annotations = []
    for ri in range(Ns_r):
        for ci in range(Np_r):
            cell_num = ri * Np_r + ci + 1
            val = grid_vals[ri, ci]
            txt = f"C{cell_num}<br>{val:.1f}" if not np.isnan(val) else f"C{cell_num}<br>ERR"
            annotations.append(dict(
                x=ci, y=ri, text=txt, showarrow=False,
                font=dict(size=11, color="white"),
                xref="x", yref="y",
            ))

    fig4 = go.Figure(go.Heatmap(
        z=grid_vals,
        x=x_labels, y=y_labels,
        colorscale=cmap,
        text=[[f"C{ri*Np_r+ci+1}\n{grid_vals[ri,ci]:.1f}"
               for ci in range(Np_r)] for ri in range(Ns_r)],
        hovertemplate="<b>%{text}</b><extra></extra>",
        colorbar=dict(title=metric_choice, len=0.8),
    ))
    fig4.update_layout(
        title=f"{Ns_r}S{Np_r}P Pack Grid — {metric_choice}",
        xaxis_title="Parallel branch →",
        yaxis_title="← Series string",
        height=max(300, Ns_r * 80 + 120),
        template="plotly_dark",
        margin=dict(t=50, b=40),
        annotations=annotations,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Imbalance metrics
    flat = grid_vals.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 1:
        ci1, ci2, ci3 = st.columns(3)
        ci1.metric(f"Min {metric_choice}", f"{flat.min():.2f}")
        ci2.metric(f"Max {metric_choice}", f"{flat.max():.2f}")
        ci3.metric(f"Imbalance (max−min)", f"{flat.max()-flat.min():.2f}")

# ── Tab 5: Degradation & SEI ──────────────────────────────────────────────────
with tab5:
    sei_cells = [r for r in good if not np.isnan(r.get("SEI_m", float("nan")))]
    if sei_cells:
        fig5 = make_subplots(rows=1, cols=2,
            subplot_titles=("SEI thickness at end (nm)", "SoH vs SEI thickness"))
        sei_vals_nm = [r["SEI_m"] * 1e9 for r in sei_cells]
        cell_ids = [r["cell_idx"] + 1 for r in sei_cells]
        soh_vals  = [r["SoH"] for r in sei_cells]
        fig5.add_trace(go.Bar(x=cell_ids, y=sei_vals_nm,
            marker_color="#fb923c", name="SEI (nm)"), row=1, col=1)
        fig5.add_trace(go.Scatter(x=sei_vals_nm, y=soh_vals,
            mode="markers", marker=dict(color="#a78bfa", size=9),
            name="SoH vs SEI"), row=1, col=2)
        fig5.update_xaxes(title_text="Cell #", row=1, col=1)
        fig5.update_xaxes(title_text="SEI thickness (nm)", row=1, col=2)
        fig5.update_yaxes(title_text="SEI (nm)", row=1, col=1)
        fig5.update_yaxes(title_text="SoH (%)", row=1, col=2)
        fig5.update_layout(height=340, template="plotly_dark",
                           margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Enable **SEI degradation** in Static Config to see degradation plots.")

    # RuL fan chart
    rul_vals = np.array([r["RuL"] for r in good if not np.isnan(r["RuL"])])
    if len(rul_vals):
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Box(
            y=rul_vals, name="RuL distribution",
            marker_color="#a78bfa", boxmean=True,
        ))
        fig_rul.add_hline(y=np.mean(rul_vals), line_dash="dash",
                          line_color="#fbbf24",
                          annotation_text=f"Mean RuL = {np.mean(rul_vals):.0f} cycles")
        fig_rul.update_layout(
            title=f"Remaining Useful Life — {Ns_r}S{Np_r}P Pack",
            yaxis_title="RuL (cycles)", template="plotly_dark",
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_rul, use_container_width=True)

# ── Tab 6: Raw data & export ──────────────────────────────────────────────────
with tab6:
    st.markdown("### Pack-level time series")
    _pn = len(pack["t"])
    def _pc(a): return flat1d_arr(a)[:_pn] if len(flat1d_arr(a)) >= _pn else np.pad(flat1d_arr(a),(0,_pn-len(flat1d_arr(a))),constant_values=flat1d_arr(a)[-1])
    pack_df = pd.DataFrame({
        "Time [s]":           pack["t"],
        "Time [h]":           pack["t"] / 3600,
        "Pack voltage [V]":   _pc(pack["V_pack"]),
        "Pack current [A]":   _pc(pack["I_pack"]),
        "Mean T [C]":         _pc(pack["T_pack"]),
        "Pack capacity [Ah]": _pc(pack["Q_pack"]),
    })
    st.dataframe(pack_df.round(5), use_container_width=True, height=250)

    buf1 = io.StringIO(); pack_df.to_csv(buf1, index=False)
    st.download_button("⬇️ Download pack time-series CSV",
        data=buf1.getvalue(), file_name="pack_timeseries.csv",
        mime="text/csv", use_container_width=True)

    st.markdown("### Per-cell summary")
    st.dataframe(cell_df.round(3), use_container_width=True, height=250)
    buf2 = io.StringIO(); cell_df.to_csv(buf2, index=False)
    st.download_button("⬇️ Download per-cell summary CSV",
        data=buf2.getvalue(), file_name="cell_summary.csv",
        mime="text/csv", use_container_width=True)

    # Cell time-series selector
    st.markdown("### Individual cell time series")
    sel_cell = st.selectbox("Select cell to inspect",
        [f"Cell {r['cell_idx']+1} (S{r['cell_idx']//Np_r+1}, P{r['cell_idx']%Np_r+1})"
         for r in good])
    sel_idx = int(sel_cell.split()[1]) - 1
    sel_r = next((r for r in good if r["cell_idx"] == sel_idx), None)
    if sel_r:
        def safe_col(arr, n):
            """Return a 1-D array of exactly length n, trimming or padding as needed."""
            a = flat1d_arr(arr)
            if len(a) > n:
                return a[:n]
            if len(a) < n:
                return np.pad(a, (0, n - len(a)), constant_values=a[-1])
            return a

        _n = len(sel_r["t"])
        cdf = pd.DataFrame({
            "Time [s]":        safe_col(sel_r["t"],      _n),
            "Voltage [V]":     safe_col(sel_r["V"],      _n),
            "Current [A]":     safe_col(sel_r["I"],      _n),
            "Capacity [Ah]":   safe_col(sel_r["Q"],      _n),
            "Temperature [C]": safe_col(sel_r["T_cell"], _n),
        })
        st.dataframe(cdf.round(5), use_container_width=True, height=200)
        buf3 = io.StringIO(); cdf.to_csv(buf3, index=False)
        st.download_button(f"⬇️ Download Cell {sel_idx+1} CSV",
            data=buf3.getvalue(), file_name=f"cell_{sel_idx+1}_timeseries.csv",
            mime="text/csv", use_container_width=True)

    # JSON summary
    summary = {
        "pack_topology": f"{Ns_r}S{Np_r}P",
        "n_cells": res["n_total"],
        "model": model_label,
        "param_set": param_set,
        "T_amb_C": res["T_amb"],
        "pack_SoC_%": round(pack["pack_soc"], 2),
        "pack_SoH_%": round(pack["pack_soh"], 2),
        "pack_RuL_cycles": round(float(res["pack_rul"]), 1),
        "pack_T_max_C": round(pack["pack_T_max"], 2),
        "thermal_gradient_C": round(pack["pack_T_max"] - pack["pack_T_min"], 3),
        "cells_simulated_OK": len(good),
        "cells_failed": len(cells) - len(good),
        "simulation_time_s": round(res["elapsed"], 2),
    }
    st.download_button("⬇️ Download summary JSON",
        data=json.dumps(summary, indent=2),
        file_name="pack_summary.json",
        mime="application/json", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Battery Pack Digital Twin · PyBaMM-powered pack estimation (SoC · SoH · RuL · ΔT) · "
    "Architecture: Dynamic Inputs → Liionpack-style Pack Core → Estimation Outputs → Feedback Loop"
)