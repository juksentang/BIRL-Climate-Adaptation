"""Statistical-disclosure scan of every git-tracked file.

LSMS-ISA microdata may not be redistributed. This repository tracks only
aggregates (country / country x action / country x asset-tercile summaries,
posterior draws of country-level parameters). Run this before committing
results:

    python3 tools/disclosure_scan.py            # exit 1 on any finding

Checks
  1. no tracked parquet / pkl / npy / GPS-point files;
  2. no tracked csv/json/npz with per-observation or per-household size
     (row or array length close to N_obs=222,023 or N_hh=15,644, or > MAX_ROWS
     unless whitelisted as a trace);
  3. no household / plot / GPS identifier fields in headers or json keys;
  4. no integer count cell below MIN_CELL in json/csv fields named like
     n_cell / n_obs / count (small-cell suppression).
"""
import csv, json, re, subprocess, sys
import numpy as np

MAX_ROWS = 5000
MIN_CELL = 10
TRACE_WHITELIST = re.compile(r"05_BIRL_SVI/results/.*_elbo\.csv$")        # (step, elbo) traces
ID_KEYS = re.compile(r"\b(hh_id\w*|household_id|plot_id|ea_id|cluster_id|lat|lon|latitude|longitude|gps_lat|gps_lon)\b", re.I)
COUNT_KEYS = re.compile(r"^(n_cell|n_obs|n_hh|n_households|n_plots|count|counts|n)$|_count$|_counts$|^n_obs_", re.I)   # data counts only, not run metadata (n_chains, n_devices, ...)

files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True).stdout.split("\0")
files = [f for f in files if f]
findings = []


def check_counts(name, obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if COUNT_KEYS.search(str(k)) and isinstance(v, (list, int)):
                arr = np.asarray(v).ravel() if isinstance(v, list) else np.asarray([v])
                if arr.dtype.kind in "iu" or (arr.dtype.kind == "f" and np.all(np.mod(arr[np.isfinite(arr)], 1) == 0)):
                    small = arr[(arr > 0) & (arr < MIN_CELL)]
                    if small.size:
                        findings.append((name, f"{path}/{k}: {small.size} count cell(s) below {MIN_CELL} (min {int(small.min())})"))
            check_counts(name, v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:500]):
            if isinstance(v, (dict, list)):
                check_counts(name, v, f"{path}[{i}]")


for f in files:
    ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
    if ext in ("parquet", "pkl", "npy", "joblib") or "gps_points" in f and ext == "csv":
        findings.append((f, "restricted file type tracked")); continue
    try:
        if ext == "csv":
            with open(f, newline="") as fh:
                header = fh.readline(); rows = sum(1 for _ in fh)
            if ID_KEYS.search(header):
                findings.append((f, "identifier column: " + header.strip()[:80]))
            if rows > MAX_ROWS and not TRACE_WHITELIST.search(f):
                findings.append((f, f"{rows} rows"))
            with open(f, newline="") as fh:
                rd = csv.DictReader(fh)
                for col in [c for c in (rd.fieldnames or []) if COUNT_KEYS.search(c)]:
                    fh.seek(0); next(fh)
                    vals = []
                    for r in csv.DictReader(fh, fieldnames=rd.fieldnames):
                        try: vals.append(float(r[col]))
                        except (TypeError, ValueError): pass
                    v = np.asarray(vals); small = v[(v > 0) & (v < MIN_CELL)]
                    if small.size and np.all(np.mod(v[np.isfinite(v)], 1) == 0):
                        findings.append((f, f"column {col}: {small.size} count(s) below {MIN_CELL}"))
        elif ext == "json":
            d = json.load(open(f))
            s = json.dumps(d)
            m = ID_KEYS.search(s)
            if m and m.group(0).lower() not in ("lat", "lon"):
                findings.append((f, f"identifier key {m.group(0)}"))
            check_counts(f, d)
        elif ext == "npz":
            d = np.load(f, allow_pickle=False)
            for k in d.files:
                if any(n > MAX_ROWS for n in d[k].shape):
                    findings.append((f, f"array {k} shape {d[k].shape}"))
    except Exception as e:
        findings.append((f, f"could not read: {e}"))

if findings:
    print("DISCLOSURE SCAN: findings")
    for f, msg in findings:
        print(f"  {f}: {msg}")
    sys.exit(1)
print(f"DISCLOSURE SCAN: clean ({len(files)} tracked files)")
