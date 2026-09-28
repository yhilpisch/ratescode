# Python & AI for Rates, Bonds, and Credit — Code & Notebooks

This repository contains the public companion notebooks, Python scripts, data
snapshots, and validation resources for *Python & AI for Rates, Bonds, and
Credit*.

## Structure

- `notebooks/` — chapter and appendix notebooks aligned with the book.
- `code/` — standalone chapter scripts and figure-generation scripts.
- `data/` — official, synthetic, and pedagogical data snapshots.
- `requirements.txt` — Python dependencies used by the examples.

## Usage

Create a Python 3.11+ environment, install the requirements, and run examples
from the repository root:

```bash
python -m pip install -r requirements.txt
python code/ch02_cash_flows_discounting.py
```

Open notebooks in JupyterLab or another notebook environment and run cells from
top to bottom. The notebooks are intentionally transparent and use relative
paths so that calculations can be inspected and adapted.

### Google Colab

The chapter and appendix notebooks link directly to Google Colab. Open a
notebook through its link and run the setup cell first. It clones this
repository into the Colab runtime, resolves the data and code paths, and checks
the core scientific Python packages. No Google Drive mount or credentials are
required. The examples use the frozen snapshots included here, not live market
feeds.

For local work, the same setup cell detects an existing checkout. Alternatively,
install `requirements.txt` and launch Jupyter from the repository root.

## Data

Official datasets include `.meta.json` sidecars. Synthetic and pedagogical CSV
files support reproducible teaching examples and should not be interpreted as
live market data.

## Disclaimer

This repository is provided for educational and illustrative purposes only and
comes without warranties or guarantees. It does not provide investment, legal,
regulatory, or risk-management advice. Do not use these examples for critical
financial decisions or production deployments without rigorous review, testing,
and validation.

## Contact

- The Python Quants: <https://tpq.io>
- CPF Program: <https://python-for-finance.com>
- Linktree: <https://linktr.ee/dyjh>
