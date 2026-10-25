# Environment

- Python 3.12 (not 3.14: SHAP and optbinning may lack wheels for 3.14).
- Created with uv, outside the repo:

```bash
~/.maki-env/bin/pip install uv
uv venv --python 3.12 ~/.fin-env
source ~/.fin-env/bin/activate
```

- Runs on CPU. GPU is not needed for P2.
- Power BI runs on Windows and cannot read from WSL2. Export results as CSV or Parquet if a dashboard is needed.
