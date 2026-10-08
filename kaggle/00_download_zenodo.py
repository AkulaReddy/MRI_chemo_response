# =============================================================================
# 00_download_zenodo.py: overnight download of BreastDCEDL archives on Kaggle
#
# HOW TO RUN (each person, tonight):
#   1. kaggle.com -> Create -> New Notebook.
#   2. Right panel "Session options":
#        Accelerator: None (no GPU needed; save your GPU quota)
#        Internet:    ON   (requires a phone-verified account)
#   3. Paste this whole file into ONE code cell. Set ME below to "A", "B" or "C".
#   4. Top right: "Save Version" -> "Save & Run All (Commit)" -> Save.
#      The run continues in the background after you close the browser (12 h limit).
#   5. Tomorrow: open the notebook -> latest version -> "Output" tab. Check the
#      log says "MD5 OK" for every file. Then "New Dataset" from the output and
#      name it exactly:
#        A -> mnj-raw-ispy1     B -> mnj-raw-ispy2     C -> mnj-raw-duke
#      Set it Private and share it with the other two (Settings -> Sharing).
#
# WHY PARALLEL PARTS: Zenodo (CERN, Geneva) limits throughput per connection
# (measured 2026-10-08: ~65-90 KB/s per connection, scaling linearly with the
# number of connections). From Kaggle one connection already gave ~1 MB/s
# (I-SPY1, 2026-10-08); parallel parts should multiply that. Each file is split into PARTS byte ranges downloaded
# at the same time, each resumable, then joined and md5-checked.
#
# Output limit on Kaggle is 20 GB in /kaggle/working. The largest job (I-SPY2,
# 13.6 GB) fits: parts are deleted as they are joined.
# =============================================================================

ME = "A"   # <-- set to "A", "B" or "C"

# Only if a previous run hit the 12 h limit before finishing: attach that run's
# output to this notebook ("+ Add Input" -> Your Work -> Notebook Output) and put
# its path here, e.g. "/kaggle/input/my-download-notebook". Its finished files
# and partial parts are copied in and the download resumes. Otherwise leave "".
RESUME_FROM = ""

PARTS = 8          # parallel connections per file; moderate, Zenodo is under load

import hashlib, json, os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor

RECORD = "18114231"
BASE = f"https://zenodo.org/records/{RECORD}/files"
JOBS = {   # (file name, md5, exact size in bytes from https://zenodo.org/api/records/18114231)
    "A": [("BreastDCEDL_ISPY1_min_crop.tar.gz", "d2d2630940a861819e6a385877f45ba5", 1252209311),
          ("BreastDCEDL_models.tar.gz",         "8096eea314a326abd284b7d95837c0d5", 319522343),
          ("BreastDCEDL_metadata_min_crop.csv", "d58b84d0f343b4b8338cbb222e6dda08", 339901)],
    "B": [("BreastDCEDL_ISPY2_min_crop.tar.gz", "2f1a00310bd00d344664319fb7bd679c", 13634158199)],
    "C": [("BreastDCEDL_DUKE_min_crop.tar.gz",  "16e6c752b9679138e88c188d7a01895f", 8643461864)],
}
# Zenodo is often overloaded and answers "504 Gateway Time-out". curl --fail makes
# any HTTP error a failure (nothing is written), and the loop backs off and retries.
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
LOG = os.path.join(OUT, "download_log.txt")
MAX_HOURS = 11.5          # stop retrying before Kaggle's 12 h limit
deadline = time.time() + MAX_HOURS * 3600


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_range(url, part_path, start, end):
    """Download bytes [start, end] into part_path, resuming; retry until complete."""
    want = end - start + 1
    failures = 0
    while time.time() < deadline:
        if os.path.exists(part_path + ".tmp"):    # bytes from an interrupted attempt/session
            with open(part_path, "ab") as dst, open(part_path + ".tmp", "rb") as src:
                shutil.copyfileobj(src, dst)
            os.remove(part_path + ".tmp")
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have == want:
            return True
        if have > want:                       # corrupt part, restart it
            os.remove(part_path); have = 0
        r = subprocess.run(
            ["curl", "-L", "-sS", "--fail", "--retry", "3", "--retry-delay", "30",
             "--speed-limit", "1024", "--speed-time", "120",
             "-r", f"{start + have}-{end}", "-o", part_path + ".tmp",
             "-w", "%{http_code}", url],
            capture_output=True, text=True)
        code = r.stdout.strip()[-3:]
        if os.path.exists(part_path + ".tmp"):
            # --fail means error pages are never written, so bytes on disk are real
            # range data, even if curl was cut off mid-transfer. The one exception:
            # a 200 (whole file) answer to a request that did not start at byte 0.
            if not (code == "200" and start + have > 0):
                with open(part_path, "ab") as dst, open(part_path + ".tmp", "rb") as src:
                    shutil.copyfileobj(src, dst)
            os.remove(part_path + ".tmp")
        if r.returncode != 0:
            failures += 1
            wait = min(60 * failures, 600)  # back off: 1, 2, ... 10 min
            if failures in (1, 5) or failures % 10 == 0:
                log(f"{os.path.basename(part_path)}: HTTP {code}, curl exit {r.returncode}, "
                    f"failure {failures}, retry in {wait}s {r.stderr.strip()[:120]}")
            time.sleep(wait)
        else:
            failures = 0
    return (os.path.getsize(part_path) if os.path.exists(part_path) else 0) == want


def download(name, expected_md5, size):
    path = os.path.join(OUT, name)
    if os.path.exists(path) and os.path.getsize(path) == size and md5sum(path) == expected_md5:
        log(f"{name}: already complete, MD5 OK")
        return True
    if os.path.exists(path):
        os.remove(path)                       # wrong size or md5: start this file over
    url = f"{BASE}/{name}?download=1"
    n = PARTS if size > 50e6 else 1
    bounds = [(i * size // n, (i + 1) * size // n - 1) for i in range(n)]
    parts = [f"{path}.part{i:02d}" for i in range(n)]
    log(f"{name}: {size/1e9:.3f} GB in {n} parts")

    def on_disk():   # finished bytes + bytes of attempts still in progress
        return sum(os.path.getsize(f) for p in parts for f in (p, p + ".tmp")
                   if os.path.exists(f))

    t0 = time.time()
    done_before = on_disk()
    with ThreadPoolExecutor(n) as ex:
        futs = [ex.submit(fetch_range, url, p, s, e) for p, (s, e) in zip(parts, bounds)]
        while not all(f.done() for f in futs):
            time.sleep(300)
            got = on_disk()
            rate = (got - done_before) / 1e6 / (time.time() - t0)
            eta = (size - got) / 1e6 / max(rate, 1e-6) / 3600
            log(f"{name}: {got/1e9:.2f}/{size/1e9:.2f} GB, {rate:.2f} MB/s, ETA {eta:.1f} h")
        ok = all(f.result() for f in futs)
    if not ok:
        log(f"{name}: INCOMPLETE at deadline; rerun with RESUME_FROM set"); return False

    with open(path, "wb") as dst:              # join, deleting parts as we go
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, 1 << 24)
            os.remove(p)
    got = md5sum(path)
    if got == expected_md5:
        log(f"{name}: MD5 OK ({got}), {(time.time()-t0)/3600:.2f} h")
        return True
    log(f"{name}: MD5 MISMATCH got {got} expected {expected_md5}; delete it and rerun")
    return False


log(f"Person {ME}: starting, {PARTS} connections per file. Free disk:\n"
    f"{subprocess.run(['df', '-h', OUT], capture_output=True, text=True).stdout}")

if RESUME_FROM:
    for f in os.listdir(RESUME_FROM):
        if f.startswith("BreastDCEDL_") and not os.path.exists(os.path.join(OUT, f)):
            # copies finished files, .partNN and .partNN.tmp alike
            shutil.copy(os.path.join(RESUME_FROM, f), os.path.join(OUT, f))
            log(f"resumed {f} from {RESUME_FROM}")

status = {}
for name, md5, size in JOBS[ME]:
    status[name] = download(name, md5, size)
with open(os.path.join(OUT, "download_status.json"), "w") as f:
    json.dump(status, f, indent=2)
log(f"DONE: {status}")
