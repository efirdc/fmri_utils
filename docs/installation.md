# Installation

`fmri_utils` requires Python 3.9+.

## Option A: Install from GitHub

```bash
pip install "git+https://github.com/efirdc/fmri_utils.git"
```

Upgrade:

```bash
pip install --upgrade --no-cache-dir "git+https://github.com/efirdc/fmri_utils.git"
```

Check installed version:

```bash
python -m pip show fmri-utils
```

## Option B: Editable install (development)

```bash
git clone https://github.com/efirdc/fmri_utils.git
cd fmri_utils
pip install -e .
```

## Optional: Create an isolated environment first

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
```
