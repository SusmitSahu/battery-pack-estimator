# 🔋 Battery Pack Digital Twin

A physics-based battery pack simulation and estimation toolbox built with **PyBaMM** and **Streamlit**. Runs entirely on a local server — no cloud dependency required.

Designed around the architecture of [liionpack](https://github.com/pybamm-team/liionpack): dynamic inputs feed a PyBaMM-powered pack core, which produces real-time estimates of **SoC**, **SoH**, **RuL**, and thermal gradients across every cell in the pack array.

---

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────┐
│   Static Config         │     │   Dynamic Inputs          │
│  (Black-Box Model)      │────▶│  I · V · T_amb · H        │
│  Materials, Geometry,   │     │  Protocol, C-rate, Cycles │
│  Pack topology (NsNp),  │     └────────────┬─────────────┘
│  Degradation params     │                  │
└─────────────────────────┘                  ▼
                                  ┌───────────────────────┐
                                  │   Pack Core            │
                                  │  (PyBaMM Solvers)      │
                                  │  Per-cell SPM/SPMe/DFN │
                                  │  + Thermal + SEI       │
                                  └────────────┬──────────┘
                                               │
                                               ▼
                                  ┌───────────────────────┐
                                  │  Estimation Outputs    │
                                  │  SoC · SoH · RuL · ΔT │
                                  └────────────┬──────────┘
                                               │
                                               ▼
                                  ┌───────────────────────┐
                                  │  Optimisation &        │
                                  │  Feedback Loop         │
                                  └───────────────────────┘
```

---

## Features

### Static Configuration (Black-Box)
| Parameter | Options |
|---|---|
| Pack topology | Any Ns × Np (e.g. 2S2P, 4S4P, up to 20S10P) |
| Electrochemical model | SPM, SPMe, DFN |
| Parameter set (chemistry) | Chen2020, Marquis2019, Ecker2015, Ai2020 |
| Thermal model | Isothermal, Lumped |
| SEI degradation | None, Constant, EC-reaction limited, Reaction limited |
| Lithium plating | Irreversible (toggle) |
| Geometry overrides | Electrode height, width |
| Heat transfer coefficient | Per-cell convective HTC (W/m²K) |
| Internal resistance offset | Contact/pack resistance per cell (Ω) |
| Cell-to-cell variation | Gaussian capacity spread σ and SoC spread σ |
| Degradation tuning | SEI growth rate factor, design cycle life |

### Dynamic Inputs
| Input | Description |
|---|---|
| Current **I** | Via C-rate (charge and discharge) |
| Voltage limits **V** | Upper and lower cut-off per cell |
| Ambient temperature **T_amb** | −20 to 60 °C slider |
| Convective HTC **H** | Per-cell heat transfer coefficient |
| Protocol | Cycling CC-CV · Constant discharge · GITT · Custom sequence |

### Estimation Outputs

- **SoC** — State of Charge per cell and pack aggregate
- **SoH** — State of Health (capacity retention %)
- **RuL** — Remaining Useful Life in cycles (linear fade model, configurable EoL at 80% SoH)
- **ΔT** — Thermal gradient across the pack (max − min cell temperature)

### Visualisation Tabs

| Tab | Content |
|---|---|
| 📈 Pack dynamics | Pack voltage V(t), current I(t), mean temperature T(t) with cell envelope band |
| 🌡️ Thermal map | All cell temperature traces, end-of-simulation distribution histogram, ΔT metric |
| 🔬 Per-cell SoC/SoH | Table + bar charts with pack mean reference lines |
| 🗺️ 2D pack grid | Spatial heatmap of any metric (SoC/SoH/RuL/T_max/SEI) in Ns × Np layout with cell labels |
| 📊 Degradation | SEI thickness per cell, SoH vs SEI scatter, RuL box plot |
| 📋 Data & Export | Pack time-series CSV, per-cell summary CSV, individual cell CSV, JSON summary |

---

## Installation

### Requirements

- Python 3.9 or later
- Anaconda or a standard virtual environment (recommended)

### Install dependencies

```bash
pip install pybamm streamlit plotly pandas scipy numpy
```

> **Note:** `pybamm` pulls in `casadi` automatically. If you want the faster `IDAKLUSolver`, install the optional dependency:
> ```bash
> pip install pybamm[odes]
> ```

### Clone and run

```bash
git clone https://github.com/your-username/battery-pack-digital-twin.git
cd battery-pack-digital-twin
streamlit run battery_pack_twin.py
```

The app opens automatically at `http://localhost:8501`.

---

## File Structure

```
battery-pack-digital-twin/
├── battery_pack_twin.py   # Main Streamlit application
└── README.md              # This file
```

---

## Quick Start

1. **Open the sidebar** — all configuration is in the left panel.
2. **Set pack topology** — choose Ns (series) and Np (parallel). Start with `2S2P` (4 cells).
3. **Choose model** — `SPM` is fastest; `DFN` gives the most accurate physics.
4. **Select chemistry** — `Chen2020` (NMC/graphite) is the most validated starting point.
5. **Set protocol** — e.g. Cycling CC-CV, 3 cycles at 1C charge / 1C discharge.
6. **Toggle physics** — enable Thermal and SEI for realistic degradation.
7. **Click ▶ Run Pack Simulation** — results appear across all tabs with live charts.
8. **Export** — download CSV or JSON from the Data & Export tab.

---

## Model Compatibility

| Model | Thermal | SEI | Li plating | Notes |
|---|---|---|---|---|
| SPM | ✅ | ✅ | ✅ | Fastest; good for large packs |
| SPMe | ✅ | ✅ | ✅ | Adds electrolyte dynamics |
| DFN | ✅ | ✅ | ✅ | Full physics; slowest |

> **Recommended combination for degradation studies:** DFN + Lumped thermal + EC reaction limited SEI + Chen2020.

> **Recommended combination for large pack sweeps:** SPM + Isothermal + Chen2020.

---

## Estimated Simulation Times

Times are approximate on a modern laptop (M-series Mac or equivalent Intel/AMD CPU).

| Pack size | Model | Thermal | Cycles | Approx. time |
|---|---|---|---|---|
| 2S2P (4 cells) | SPM | Isothermal | 3 | ~10 s |
| 2S2P (4 cells) | DFN | Lumped | 3 | ~45 s |
| 4S4P (16 cells) | SPM | Lumped | 3 | ~60 s |
| 4S4P (16 cells) | DFN | Lumped | 1 | ~3 min |

---

## Estimation Methods

### State of Charge (SoC)
Computed from the ratio of discharged capacity to maximum observed capacity over the simulation window.

```
SoC = (1 − Q_discharged / Q_max) × 100 %
```

### State of Health (SoH)
Capacity retention relative to the nominal cell capacity defined in the static configuration.

```
SoH = (Q_actual / Q_nominal) × 100 %
```

### Remaining Useful Life (RuL)
Linear fade model from 100% SoH at cycle 0 to end-of-life (80% SoH) at the user-defined design cycle life.

```
RuL = (SoH − 80%) / fade_rate_per_cycle
```

### Thermal gradient (ΔT)
```
ΔT = T_cell_max − T_cell_min   [°C, end of simulation]
```

---

## Known Limitations

- **Li plating** requires `Chen2020` and `DFN`; using it with other combinations will raise a PyBaMM error displayed inline.
- **MSMR / MPM** models are not included; the toolbox covers SPM, SPMe, and DFN.
- **Pack simulation** runs cells sequentially (not in parallel threads). Large packs with DFN can take several minutes.
- **SoH and RuL** are simplified estimates. For high-fidelity degradation, run multi-cycle experiments with SEI enabled.
- Cell-to-cell variation seeds are fixed (`numpy.random.default_rng(42)`) for reproducibility. Change the seed in `run_single_cell` if needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| [PyBaMM](https://www.pybamm.org) | Electrochemical cell simulation |
| [Streamlit](https://streamlit.io) | Web front-end |
| [Plotly](https://plotly.com/python/) | Interactive charts and heatmaps |
| [Pandas](https://pandas.pydata.org) | Data tables and CSV export |
| [SciPy](https://scipy.org) | Interpolation for pack aggregation |
| [NumPy](https://numpy.org) | Numerical operations |

---

## Contributing

Pull requests are welcome. For major changes please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

- Doyle, M., Fuller, T. F., & Newman, J. (1993). Modeling of Galvanostatic Charge and Discharge of the Lithium/Polymer/Insertion Cell. *Journal of the Electrochemical Society*, 140(6), 1526.
- Chen, C. H., et al. (2020). Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models. *Journal of the Electrochemical Society*, 167(8), 080534.
- Sulzer, V., et al. (2021). Python Battery Mathematical Modelling (PyBaMM). *Journal of Open Research Software*, 9(1).
- Python Battery Mathematical Modelling (PyBaMM): [https://www.pybamm.org](https://www.pybamm.org)
- liionpack — Pack-level simulation with PyBaMM: [https://github.com/pybamm-team/liionpack](https://github.com/pybamm-team/liionpack)
