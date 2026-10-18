"""Download the FULL BreastDCEDL release (Zenodo record 18114231) with resume and md5 checks.

    python scripts/download_breastdcedl.py                         # everything -> datasets/BreastDCEDL/
    python scripts/download_breastdcedl.py --files ispy2           # one archive
    python scripts/download_breastdcedl.py --out /kaggle/working/BreastDCEDL --files ispy1 models meta

The three cohort archives together hold all 2,070 patients (I-SPY1 172, I-SPY2 982, Duke 916); nothing is
sub-sampled. Each file is fetched as PARTS parallel byte ranges (Zenodo throttles single connections), every
part is resumed until it has exactly the right size, then the parts are joined and checked against the md5
that Zenodo publishes. Rerunning resumes; finished, verified files are skipped. Needs `curl` (macOS, Linux and
Kaggle have it). No login, licence CC BY-NC 4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://zenodo.org/records/18114231/files"
FILES = {   # key: (file name, exact size in bytes, md5) from https://zenodo.org/api/records/18114231
    "ispy1":  ("BreastDCEDL_ISPY1_min_crop.tar.gz", 1252209311, "d2d2630940a861819e6a385877f45ba5"),
    "ispy2":  ("BreastDCEDL_ISPY2_min_crop.tar.gz", 13634158199, "2f1a00310bd00d344664319fb7bd679c"),
    "duke":   ("BreastDCEDL_DUKE_min_crop.tar.gz", 8643461864, "16e6c752b9679138e88c188d7a01895f"),
    "models": ("BreastDCEDL_models.tar.gz", 319522343, "8096eea314a326abd284b7d95837c0d5"),
    "meta":   ("BreastDCEDL_metadata_min_crop.csv", 339901, "d58b84d0f343b4b8338cbb222e6dda08"),
    "demo":   ("BreastDCEDL_demo_data.tar.gz", 54838099, "5b7d8701309c2c3807f8d126f7ddcede"),
}
FULL = ["ispy1", "ispy2", "duke", "models", "meta"]   # the full release (demo is a 7-patient sample)


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(key, out_dir: Path, parts=8, max_hours=11.5):
    name, size, md5 = FILES[key]
    out = out_dir / name
    if out.exists() and out.stat().st_size == size:
        if md5sum(out) == md5:
            print(f"{name}: already complete, md5 OK", flush=True)
            return True
        out.unlink()
    url, step, t0 = f"{BASE}/{name}?download=1", -(-size // parts), time.time()
    deadline = t0 + max_hours * 3600

    def part(i):
        a, b = i * step, min(size, (i + 1) * step) - 1
        p = out_dir / f"{name}.part{i}"
        tries = 0
        while time.time() < deadline:
            have = p.stat().st_size if p.exists() else 0
            if have > b - a + 1:                      # corrupt part: start it again
                p.unlink()
                continue
            if have == b - a + 1:
                return True
            with open(p, "ab") as f:                  # --fail: an HTTP error (Zenodo 504) writes nothing
                subprocess.run(["curl", "-sL", "--fail", "-m", "1800", "-r", f"{a + have}-{b}", url], stdout=f)
            tries += 1
            time.sleep(min(60, 2 * tries))
        return False

    print(f"{name}: {size / 1e9:.2f} GB in {parts} parts ...", flush=True)
    with ThreadPoolExecutor(parts) as ex:
        futs = [ex.submit(part, i) for i in range(parts)]
        while not all(f.done() for f in futs):
            time.sleep(60)
            got = sum((out_dir / f"{name}.part{i}").stat().st_size for i in range(parts)
                      if (out_dir / f"{name}.part{i}").exists())
            el = time.time() - t0
            print(f"  {name}: {got / 1e9:.2f}/{size / 1e9:.2f} GB, {got / max(el, 1) / 1e6:.2f} MB/s", flush=True)
        if not all(f.result() for f in futs):
            print(f"{name}: time limit reached; rerun to resume", flush=True)
            return False
    with open(out, "wb") as f:
        for i in range(parts):
            p = out_dir / f"{name}.part{i}"
            with open(p, "rb") as src:
                while (b := src.read(1 << 24)):
                    f.write(b)
            p.unlink()
    ok = md5sum(out) == md5
    dt = time.time() - t0
    print(f"{name}: {size / 1e6 / dt:.2f} MB/s average, md5 {'OK' if ok else 'MISMATCH (deleted, rerun)'}", flush=True)
    if not ok:
        out.unlink()
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: datasets/BreastDCEDL/ in this repo")
    ap.add_argument("--files", nargs="+", default=FULL, choices=list(FILES))
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=11.5, help="stop before Kaggle's 12 h session limit")
    a = ap.parse_args(argv)
    out = Path(a.out or Path(__file__).resolve().parents[1] / "datasets" / "BreastDCEDL")   # __file__: not in notebooks
    out.mkdir(parents=True, exist_ok=True)
    need = sum(FILES[k][1] for k in a.files)
    free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
    print(f"downloading {a.files} ({need / 1e9:.1f} GB) into {out} (free disk {free / 1e9:.1f} GB)", flush=True)
    if free < need * 1.05:
        print("WARNING: not enough free disk for these files")
    status = {k: fetch(k, out, a.parts, a.max_hours) for k in a.files}
    (out / "download_status.json").write_text(json.dumps(
        {FILES[k][0]: ("md5 OK" if v else "INCOMPLETE") for k, v in status.items()}, indent=2))
    print("\nSUMMARY:", "ALL MD5 OK" if all(status.values()) else "some files incomplete: rerun the same command")
    for k, v in status.items():
        print(f"  {'OK ' if v else '-- '} {FILES[k][0]}")
    return all(status.values())


if __name__ == "__main__":
    main()
