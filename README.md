# CIGP Autocall Pricer

This project contains a comprehensive Autocall Athena pricer, exploring zero-coupon (ZC) bootstrapping, path-dependent digitals, put down-and-in (PDI), worst-of features, and decrement indices, complete with a Streamlit front-end.

## Getting Started

Follow these instructions to go from a fresh clone to a working pipeline.

### Prerequisites

- You must have **Python 3.14** installed on your system.

### 1. Install `uv`
This project relies on `uv` for lightning-fast dependency and virtual environment management.
If you haven't installed `uv` yet, you can do so by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(Windows users can install via PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)*

### 2. Clone the Repository
Clone the project repository to your local machine:

```bash
git clone <your-repo-url>
cd cigp-autocall-pricer
```

### 3. Sync the Environment
Use `uv` to automatically create a virtual environment, resolve dependencies, and lock them exactly as specified in `uv.lock`:

```bash
uv sync
```

### 4. Run the Pipeline
Once the environment is synced, you can execute the main script using `uv run`, which handles everything inside the isolated environment:

```bash
uv run python main.py
```

## External Deployment (Streamlit Cloud)

This project is configured for one-click deployment to **Streamlit Community Cloud**.

### Permanent Link
Once deployed, the app will be accessible at: `https://your-app-name.streamlit.app`

### Maintenance
To update the app, simply push your changes to the `main` branch of your GitHub repository. Streamlit Cloud will automatically detect the changes and redeploy.

---

*Project maintained by CIGP HK Quantitative Team.*

