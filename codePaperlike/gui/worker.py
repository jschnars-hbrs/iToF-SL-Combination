"""One GUI job (calibration or test run) in its own process, so it always runs the current code.

    python -m gui.worker <run_dir>/job.json

Progress goes to stdout as normal prints plus GUI_EVENT|{json} lines the GUI listens for.
Figures are saved as PNG and pickled matplotlib figures, results as results.json, all inside the run folder.
"""

import contextlib
import io
import json
import pickle
import re
import subprocess
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import approaches  # noqa: E402
import calibrate  # noqa: E402
import reconstruction  # noqa: E402
import run_all  # noqa: E402
import sl_tilt_correction  # noqa: E402
from dot_calibration import DotCalibration, plot_sor_effect  # noqa: E402


def event(**kw):
    """Machine-readable progress line for the GUI."""
    print("GUI_EVENT|" + json.dumps(kw, default=str), flush=True)


def safe(name):
    """File-system safe folder name."""
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "scene"


def save_figures(figures, out_dir):
    """PNG + pickle of every (name, fig); returns the folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        if fig is None:
            continue
        fig.savefig(str(out_dir / f"{name}.png"), dpi=130, bbox_inches="tight")
        try:
            with open(out_dir / f"{name}.fig.pkl", "wb") as f:
                pickle.dump(fig, f)
        except Exception as exc:
            print(f"  (figure {name} not interactive: {exc})")
        plt.close(fig)
    return out_dir


def apply_class_overrides(overrides):
    """DotCalibration class attributes (SOR, z range, EXR channel, ...) from the GUI."""
    for name, value in overrides.items():
        if isinstance(value, list):
            value = tuple(value)
        setattr(DotCalibration, name, value)
        print(f"(override DotCalibration.{name} = {value!r})")


def run_calibration(job, run_dir):
    """Compute and save a calibration (calibrate.calibrate + save_calibration)."""
    c = job["calibrate"]
    cfg = calibrate.CAMERAS[c["camera"]]
    for key in ("sl_cal", "tof_cal"):
        if c.get(key):
            cfg[key] = Path(c[key])
    dotCal = DotCalibration()
    cal = calibrate.calibrate(c["camera"], dotCal, **c["kwargs"])
    figures = calibrate.calibration_figures(dotCal, cal) if c.get("plots", True) else []
    for name, fig in figures:
        fig.canvas.draw()
    cal_file = calibrate.save_calibration(cal, c["name"], figures)
    save_figures(figures, run_dir / "figures")
    event(type="figures", scene="Kalibrierung", dir=run_dir / "figures")
    event(type="calibration", path=cal_file)


def convert_raw(scene, raw_cfg, cal_file, out_dir):
    """RAW frames → mean PCD via itof_raw_bridge.py in the itofcameracalibration environment."""
    if not raw_cfg.get("python") or not raw_cfg.get("repo") or not raw_cfg.get("kalibration"):
        raise RuntimeError("RAW input needs the itofcameracalibration Python, repository and HDF5 calibration "
                           "(Einstellungen)")
    cmd = [raw_cfg["python"], str(CODE_DIR / "itof_raw_bridge.py"), "--repo", raw_cfg["repo"],
           "--raw", scene["raw"], "--kalibration", raw_cfg["kalibration"], "--dot_calibration", str(cal_file),
           "--mode", raw_cfg.get("mode", "auto"), "--out", str(out_dir), "--name", safe(scene["name"]),
           "--params", json.dumps(raw_cfg.get("params", {}))]
    if raw_cfg.get("frames"):
        cmd += ["--frames", str(int(raw_cfg["frames"]))]
    if raw_cfg.get("each"):
        cmd.append("--each")
    print(f"RAW → PCD: {scene['raw']}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    pcd = None
    for line in proc.stdout:
        if line.startswith("BRIDGE_PCD|"):
            pcd = line.split("|", 1)[1].strip()
        else:
            print("  " + line.rstrip(), flush=True)
    if proc.wait() != 0 or pcd is None:
        raise RuntimeError(f"RAW conversion failed (exit code {proc.returncode})")
    return pcd


def setup_figure(setup, name):
    """Test SL image with the calibration trails (as in the notebook)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(setup["sl_gray"], cmap="gray")
    for t in setup["trails"]:
        if t is not None:
            ax.plot(t["u"], t["v"], lw=0.6, alpha=0.7, color="cyan")
    ax.set_title(f"{name}: SL image with calibration trails")
    ax.axis("off")
    return fig


def run_scene(scene, job, dotCal, cal, run_dir):
    """All selected approaches on one scene; returns the result rows."""
    t = job["test"]
    out = run_dir / safe(scene["name"])
    out.mkdir(parents=True, exist_ok=True)

    tof = scene.get("tof")
    if scene.get("raw"):
        tof = convert_raw(scene, t.get("raw", {}), t["calibration"], out / "tof")
    event(type="pcd", scene=scene["name"], path=tof, label=f"{scene['name']} · ToF")

    approaches.GT_DISTANCE = scene.get("gt")
    setup = approaches.common_setup(dotCal, cal, scene.get("sl"), tof)
    setup["_output_dir"] = out
    figures = []
    if t.get("plots", True):
        figures.append(("00_setup", setup_figure(setup, scene["name"])))
        if dotCal.sor:
            figures.append(("00_sor_effect", plot_sor_effect(dotCal, tof)))

    rows, results = [], {}
    for n in t["approaches"]:
        print(f"\n{'=' * 60}\n  {scene['name']} · Approach {n}\n{'=' * 60}\n", flush=True)
        res = approaches.RUNNERS[n](dotCal, cal, setup, make_plots=t.get("plots", True))
        results[n] = res
        figures += list(res["figures"].items())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            approaches.print_average_error(n, res["depths"])
        print(buf.getvalue(), end="", flush=True)
        err = run_all.parse_avg_err_lines(buf.getvalue(), scene["name"], "")
        n_valid, n_total, mean_z, std_z = approaches.depth_stats(res["depths"])
        row = {"scene": scene["name"], "approach": n, "n_valid": n_valid, "n_total": n_total,
               "mean_z_m": mean_z, "std_z_m": std_z}
        if err:
            row.update({k: err[0][k] for k in ("gt_m", "mean_m", "abs_mean_m", "std_m", "min_m", "max_m")})
        rows.append(row)
        event(type="result", row=row)

    if t.get("export") and results:
        pcd_dir = out / "pcd"
        before = set(pcd_dir.glob("*.pcd")) if pcd_dir.exists() else set()
        reconstruction.export_pcds(results, cal["K"], setup["H"], setup["W"], out_dir=pcd_dir,
                                   stem=safe(scene["name"]), **t.get("export_kwargs", {}))
        for p in sorted(set(pcd_dir.glob("*.pcd")) - before):
            event(type="pcd", scene=scene["name"], path=p, label=f"{scene['name']} · {p.stem}")

    save_figures(figures, out / "figures")
    event(type="figures", scene=scene["name"], dir=out / "figures")
    return rows


def run_test(job, run_dir):
    """Every enabled scene with the selected calibration."""
    t = job["test"]
    dotCal = DotCalibration()
    cal = calibrate.load_saved_calibration(dotCal, name=str(Path(t["calibration"]).parent),
                                           light_path_correction=t.get("light_path_correction", False))
    cal = sl_tilt_correction.correct_cal(cal, **t.get("tilt", {}))
    all_rows = []
    scenes = t["scenes"]
    for i, scene in enumerate(scenes):
        event(type="scene_start", scene=scene["name"], i=i, n=len(scenes))
        print(f"\n{'#' * 70}\n  [{i + 1}/{len(scenes)}] {scene['name']}\n{'#' * 70}\n", flush=True)
        try:
            all_rows += run_scene(scene, job, dotCal, cal, run_dir)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            event(type="scene_failed", scene=scene["name"])
    with open(run_dir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=1, default=float)


def main():
    job_file = Path(sys.argv[1])
    job = json.loads(job_file.read_text())
    run_dir = job_file.parent
    apply_class_overrides(job.get("class_overrides", {}))
    try:
        {"calibrate": run_calibration, "test": run_test}[job["kind"]](job, run_dir)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        event(type="failed")
        sys.exit(1)
    event(type="done")


if __name__ == "__main__":
    main()
