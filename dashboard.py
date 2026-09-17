#!/usr/bin/env python3
"""Live training dashboard for a model_v7 run.

Reads only files the training process already writes, so it cannot slow the run
down or fail it. Standard library only.

    python dashboard.py <run_output_dir> [--port 8088]

Then open http://localhost:8088 on Windows (WSL2 forwards localhost).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REFRESH_SECONDS = 20

# Curves worth a chart, in the order they are drawn.
CHARTS = (
    ("loss", "val_loss", "Total loss"),
    ("semantic_loss", "val_semantic_loss", "Semantic loss"),
    # V7 trains one instance-geometry head, not two. Plotting centre and offset
    # here showed empty panels and hid the only curve that says whether instance
    # separation is being learned at all.
    ("inner_distance_loss", "val_inner_distance_loss", "Inner-distance loss"),
    ("boundary_loss", "val_boundary_loss", "Boundary loss"),
    ("semantic_foreground_miou", "val_semantic_foreground_miou", "Foreground mIoU"),
)

# Parameters worth showing, grouped. Keys are looked up in training_config.json
# at any nesting depth.
PARAM_GROUPS = (
    ("Run", ("training_mode", "epochs", "seed", "output_dir", "array_dir")),
    ("Optimisation", ("batch_size", "gradient_accumulation_steps",
                      "effective_batch_size", "learning_rate",
                      "min_learning_rate", "weight_decay", "warmup_epochs",
                      "use_ema", "ema_momentum", "optimizer")),
    ("Precision", ("mixed_precision_policy", "steps_per_execution",
                   "jit_compile")),
    ("Loss weights", ("semantic", "center", "offset", "boundary")),
    ("Augmentation", ("augmentation_cycle_length", "use_dihedral_augmentation",
                      "use_copy_paste_augmentation", "copy_paste_probability",
                      "zoom_out_range", "zoom_out_probability")),
    ("Selection", ("instance_selection_metric", "instance_evaluation_iou",
                   "early_stopping_patience")),
)


# Columns InstanceF1Checkpoint contributes only on the epochs it evaluates.
# Keras re-derives the CSV column set after every resume, so a resume onto a
# non-evaluating epoch writes rows narrower than the header. csv.DictReader
# maps positionally and would shift every later value onto the wrong name, so
# short rows are matched against the header with these names removed - which
# is exactly the order Keras wrote them in.
OPTIONAL_COLUMNS = frozenset({
    "val_instance_precision", "val_instance_recall", "val_instance_f1",
    "val_instance_mask_map50", "val_instance_mask_map50_95",
    "val_instance_checkpoint_semantic_miou", "val_instance_selection_score",
})


def read_log(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return []
            narrow = [n for n in header if n not in OPTIONAL_COLUMNS]
            for values in reader:
                if not values or values[0] == header[0]:
                    continue
                if len(values) == len(header):
                    names = header
                elif len(values) == len(narrow):
                    names = narrow
                else:
                    continue
                row: dict[str, float] = {}
                for key, value in zip(names, values):
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if number == number and abs(number) != float("inf"):
                        row[key] = number
                if row:
                    rows.append(row)
    except OSError:
        return []
    rows.sort(key=lambda r: r.get("epoch", 0.0))
    return rows


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def flatten(value, out: dict[str, object], prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, out, str(key))
    elif prefix:
        out.setdefault(prefix, value)
    return out


def gpu_status() -> list[dict[str, str]]:
    fields = ("name", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "clocks.sm")
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append(dict(zip(fields, parts)))
    return rows


def sparkline(rows, train_key, val_key, title, width=460, height=150):
    train = [(r["epoch"] + 1, r[train_key]) for r in rows
             if "epoch" in r and train_key in r]
    val = [(r["epoch"] + 1, r[val_key]) for r in rows
           if "epoch" in r and val_key in r]
    if not train and not val:
        return ""
    points = train + val
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(max(xs), min(xs) + 1)
    y_lo, y_hi = min(ys), max(ys)
    if y_hi - y_lo < 1e-12:
        y_hi = y_lo + 1e-12
    pad_l, pad_r, pad_t, pad_b = 52, 10, 26, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def project(x, y):
        px = pad_l + (x - x_lo) / (x_hi - x_lo) * plot_w
        py = pad_t + (1.0 - (y - y_lo) / (y_hi - y_lo)) * plot_h
        return f"{px:.1f},{py:.1f}"

    def path(series):
        return " ".join(project(x, y) for x, y in series)

    grid = []
    for fraction in (0.0, 0.5, 1.0):
        y = y_lo + fraction * (y_hi - y_lo)
        py = pad_t + (1.0 - fraction) * plot_h
        grid.append(
            f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width - pad_r}" '
            f'y2="{py:.1f}" class="grid"/>'
            f'<text x="{pad_l - 6}" y="{py + 3.5:.1f}" class="tick" '
            f'text-anchor="end">{y:.4g}</text>'
        )
    last_bits = []
    if train:
        last_bits.append(f'<tspan class="k-train">train {train[-1][1]:.4g}</tspan>')
    if val:
        last_bits.append(f'<tspan class="k-val">val {val[-1][1]:.4g}</tspan>')
    return f"""
<figure class="chart">
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    {''.join(grid)}
    <polyline class="train" points="{path(train)}"/>
    <polyline class="val" points="{path(val)}"/>
    <text x="{pad_l}" y="14" class="ct">{html.escape(title)}</text>
    <text x="{width - pad_r}" y="14" class="cv" text-anchor="end">
      {' · '.join(last_bits)}</text>
    <text x="{width - pad_r}" y="{height - 6}" class="tick"
      text-anchor="end">epoch {int(x_hi)}</text>
  </svg>
</figure>"""


def instance_table(history) -> str:
    """Render instance_checkpoint_history.json.

    Its `epoch` is already 1-based, and its keys are the bare metric names
    (`f1`, `precision`, `recall`) rather than the `val_instance_*` names the
    same numbers get in the CSV log.
    """
    if not isinstance(history, list) or not history:
        return ("<p class='muted'>No instance evaluation recorded yet — the "
                "first runs at epoch 1, then every 5 epochs.</p>")
    records = [r for r in history if isinstance(r, dict)]
    if not records:
        return "<p class='muted'>No usable instance evaluation records.</p>"
    columns = (
        ("epoch", "Epoch", "{:.0f}"),
        ("mask_map50_95", "mAP50-95", "{:.4f}"),
        ("mask_map50", "mAP50", "{:.4f}"),
        ("f1", "F1@.50", "{:.4f}"),
        ("precision", "Precision", "{:.4f}"),
        ("recall", "Recall", "{:.4f}"),
        ("semantic_foreground_miou", "FG mIoU", "{:.4f}"),
        ("images_evaluated", "Images", "{:.0f}"),
    )
    scores = [r["mask_map50_95"] for r in records
              if isinstance(r.get("mask_map50_95"), (int, float))]
    best = max(scores) if scores else None
    body = []
    for record in reversed(records[-12:]):
        cells = []
        for key, _, fmt in columns:
            value = record.get(key)
            cells.append(fmt.format(value)
                         if isinstance(value, (int, float)) else "—")
        mark = (" class='best'"
                if best is not None and record.get("mask_map50_95") == best
                else "")
        body.append(f"<tr{mark}>" + "".join(f"<td>{c}</td>" for c in cells)
                    + "</tr>")
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{label}</th>" for _, label, _ in columns)
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def render(run_dir: Path) -> str:
    rows = read_log(run_dir / "training_log.csv")
    config = read_json(run_dir / "training_config.json") or {}
    # The file the InstanceF1Checkpoint callback actually writes.
    history = read_json(run_dir / "instance_checkpoint_history.json")
    if history is None:
        history = read_json(run_dir / "instance_evaluation_history.json")
    if isinstance(history, dict):
        history = (history.get("evaluations") or history.get("history")
                   or history.get("records") or [])
    flat = flatten(config, {})

    total_epochs = int(flat.get("epochs") or 0)
    done = len(rows)
    latest = rows[-1] if rows else {}

    # Epoch pace from the log file's own mtime history is unreliable; use the
    # elapsed wall time since the first row was written instead.
    log_path = run_dir / "training_log.csv"
    pace = eta = "—"
    if done >= 2 and log_path.is_file():
        try:
            started = log_path.stat().st_ctime
            elapsed = time.time() - started
            per_epoch = elapsed / done
            pace = f"{per_epoch / 60:.1f} min/epoch"
            if total_epochs > done:
                remaining = timedelta(seconds=int(per_epoch * (total_epochs - done)))
                finish = datetime.now() + remaining
                eta = f"{remaining} (≈ {finish:%a %d %b %H:%M})"
        except OSError:
            pass

    progress = (done / total_epochs * 100.0) if total_epochs else 0.0

    param_html = []
    for group, keys in PARAM_GROUPS:
        items = []
        for key in keys:
            if key in flat:
                value = flat[key]
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(v) for v in value)
                items.append(
                    f"<div class='p'><dt>{html.escape(key.replace('_',' '))}</dt>"
                    f"<dd>{html.escape(str(value))}</dd></div>"
                )
        if items:
            param_html.append(
                f"<section class='pg'><h3>{html.escape(group)}</h3>"
                f"<dl>{''.join(items)}</dl></section>"
            )

    charts = "".join(sparkline(rows, a, b, t) for a, b, t in CHARTS)

    gpu_html = []
    for gpu in gpu_status():
        try:
            used = float(gpu["memory.used"])
            total = float(gpu["memory.total"])
            fraction = used / total * 100.0
        except (KeyError, ValueError, ZeroDivisionError):
            used = total = fraction = 0.0
        gpu_html.append(f"""
        <div class="gpu">
          <strong>{html.escape(gpu.get('name', 'GPU'))}</strong>
          <div class="bar"><span style="width:{min(fraction,100):.0f}%"></span></div>
          <span class="muted">{used:.0f} / {total:.0f} MiB ·
            {html.escape(gpu.get('utilization.gpu','?'))}% util ·
            {html.escape(gpu.get('temperature.gpu','?'))}°C ·
            {html.escape(gpu.get('power.draw','?'))}/{html.escape(gpu.get('power.limit','?'))} W ·
            {html.escape(gpu.get('clocks.sm','?'))} MHz</span>
        </div>""")

    def metric(key, label, fmt="{:.4f}"):
        value = latest.get(key)
        shown = fmt.format(value) if isinstance(value, (int, float)) else "—"
        return f"<div class='kpi'><span>{label}</span><strong>{shown}</strong></div>"

    best_map = "—"
    if isinstance(history, list) and history:
        values = [r["mask_map50_95"] for r in history
                  if isinstance(r, dict)
                  and isinstance(r.get("mask_map50_95"), (int, float))]
        if values:
            best_map = f"{max(values):.4f}"

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<title>{html.escape(run_dir.name)} — training</title>
<style>
 :root {{ color-scheme: dark; --bg:#0f1115; --panel:#171a21; --line:#252a34;
          --fg:#e6e9ef; --muted:#8b93a3; --train:#5aa9e6; --val:#f2a65a;
          --good:#5fd08a; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--fg);
   font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
 header {{ padding:18px 24px; border-bottom:1px solid var(--line);
   display:flex; flex-wrap:wrap; gap:18px; align-items:baseline; }}
 h1 {{ font-size:17px; margin:0; font-weight:600; }}
 h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.08em;
   color:var(--muted); margin:28px 0 10px; }}
 h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em;
   color:var(--muted); margin:0 0 8px; font-weight:600; }}
 main {{ padding:0 24px 48px; max-width:1400px; }}
 .muted {{ color:var(--muted); }}
 .progress {{ height:6px; background:var(--line); border-radius:3px; overflow:hidden;
   margin:14px 0 4px; }}
 .progress span {{ display:block; height:100%; background:var(--good); }}
 .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
   gap:10px; margin-top:14px; }}
 .kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
   padding:10px 12px; }}
 .kpi span {{ display:block; color:var(--muted); font-size:11px;
   text-transform:uppercase; letter-spacing:.05em; }}
 .kpi strong {{ font-size:19px; font-variant-numeric:tabular-nums; }}
 .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
   gap:12px; }}
 .chart {{ margin:0; background:var(--panel); border:1px solid var(--line);
   border-radius:8px; padding:6px; }}
 svg {{ width:100%; height:auto; display:block; }}
 polyline {{ fill:none; stroke-width:1.8; vector-effect:non-scaling-stroke; }}
 polyline.train {{ stroke:var(--train); }}
 polyline.val {{ stroke:var(--val); }}
 .grid {{ stroke:var(--line); stroke-width:1; }}
 .tick {{ fill:var(--muted); font-size:9px; }}
 .ct {{ fill:var(--fg); font-size:11px; font-weight:600; }}
 .cv {{ font-size:10px; }} .k-train {{ fill:var(--train); }} .k-val {{ fill:var(--val); }}
 .params {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
   gap:12px; }}
 .pg {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
   padding:12px 14px; }}
 dl {{ margin:0; }} .p {{ display:flex; justify-content:space-between; gap:12px;
   padding:3px 0; border-bottom:1px solid var(--line); }}
 .p:last-child {{ border-bottom:0; }}
 dt {{ color:var(--muted); }} dd {{ margin:0; font-variant-numeric:tabular-nums;
   text-align:right; word-break:break-all; }}
 table {{ width:100%; border-collapse:collapse; background:var(--panel);
   border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
 th,td {{ padding:7px 10px; text-align:right; font-variant-numeric:tabular-nums;
   border-bottom:1px solid var(--line); }}
 th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;
   letter-spacing:.05em; text-align:right; }}
 tr.best td {{ color:var(--good); font-weight:600; }}
 .gpu {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
   padding:12px 14px; margin-bottom:8px; }}
 .bar {{ height:6px; background:var(--line); border-radius:3px; overflow:hidden;
   margin:8px 0 6px; }}
 .bar span {{ display:block; height:100%; background:var(--train); }}
</style></head><body>
<header>
  <h1>{html.escape(str(flat.get('training_mode', 'training')))} — {html.escape(run_dir.name)}</h1>
  <span class="muted">epoch {done} / {total_epochs or '?'} · {pace} · ETA {html.escape(eta)}</span>
  <span class="muted" style="margin-left:auto">refreshes every {REFRESH_SECONDS}s ·
    {datetime.now():%H:%M:%S}</span>
</header>
<main>
  <div class="progress"><span style="width:{progress:.2f}%"></span></div>
  <div class="kpis">
    {metric('loss', 'train loss')}
    {metric('val_loss', 'val loss')}
    {metric('semantic_foreground_miou', 'train FG mIoU')}
    {metric('val_semantic_foreground_miou', 'val FG mIoU')}
    <div class="kpi"><span>best mask mAP50-95</span><strong>{best_map}</strong></div>
    {metric('learning_rate', 'learning rate', '{:.2e}')}
  </div>

  <h2>Instance evaluation (validation)</h2>
  {instance_table(history)}

  <h2>Curves</h2>
  <div class="charts">{charts or "<p class='muted'>waiting for epoch 1…</p>"}</div>

  <h2>Hardware</h2>
  {''.join(gpu_html) or "<p class='muted'>nvidia-smi unavailable</p>"}

  <h2>Training parameters</h2>
  <div class="params">{''.join(param_html) or "<p class='muted'>training_config.json not written yet</p>"}</div>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    run_dir = Path(".")

    def do_GET(self):  # noqa: N802
        try:
            page = render(self.run_dir).encode("utf-8")
        except Exception as error:  # a dashboard must never take the run down
            page = (
                f"<pre>dashboard error: {html.escape(repr(error))}</pre>"
                f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>"
            ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--port", type=int, default=8088)
    arguments = parser.parse_args()
    Handler.run_dir = arguments.run_dir
    server = ThreadingHTTPServer(("0.0.0.0", arguments.port), Handler)
    print(f"dashboard for {arguments.run_dir} on http://localhost:{arguments.port}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
