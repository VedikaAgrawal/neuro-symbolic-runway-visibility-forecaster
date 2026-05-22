# Climate Visibility Forecasting Engine (Delhi-NCR Space)
An M.Tech level research-grade spatial-temporal data engineering and predictive framework built to ingest, fuse, and model multi-station meteorological parameters alongside NASA AERONET aerosol optical depth (AOD) characteristics to forecast severe visibility degradation events.

---

## 🏗️ Project Architecture
```text
climate-visibility-new/
├── data/
│   ├── raw/                 # Unaltered ASCII ISD files & NASA AERONET CSVs
│   └── processed/           # Final fused, hourly synchronized spatial matrix
├── docs/
│   └── dataset_reference.md # Complete column metadata, byte offsets, and decoding manuals
├── notebooks/               # Jupyter Engineering Workbooks
│   └── 01_data_fusion_lake.ipynb
├── pyproject.toml           # Poetry stack configuration and explicit definitions
├── poetry.lock              # Complete cryptographic package dependency lockfile
└── README.md                # Environment onboarding & pipeline trigger runbook
⚡ Environment Onboarding & Setup (Poetry)
This project leverages Poetry for isolated, deterministic, production-grade virtual environment and dependency layer management. Follow these steps to provision your machine (optimized for Apple Silicon / M1 MacBooks).

1. Clone & Enter Repository Context
Open your terminal window inside VS Code and verify your current directory path:

Bash
cd ~/Documents/climate-visibility-new
2. Configure Poetry to Nest Virtual Environments
Force Poetry to compile and place the physical binary .venv folder locally right inside your project directory (crucial for seamless VS Code Jupyter Kernel discovery):

Bash
poetry config virtualenvs.in-project true
3. Provision Isolated Stack Dependencies
Read the cryptographic poetry.lock definitions and install all core engineering and modeling packages inside your local environment sandbox:

Bash
poetry install
This instantly pulls and locks verified versions of pandas, numpy, scikit-learn, matplotlib, re, jupyter, and database driver extensions.