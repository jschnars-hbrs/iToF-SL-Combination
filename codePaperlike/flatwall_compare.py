"""Flat-wall test: per-dot errors of Approach 1/2/3 for 10x10 vs. 33x33 in one comparison figure.

    python flatwall_compare.py [--runs ../Results/flatwall_6.11_v2] [--ref ../Results/flatwall_6.11]

Reads the GUI test runs (`<runs>/<illumination>/job.json` + the PCDs `gui/worker.py` wrote into `<scene>/tof/`),
re-runs the approaches exactly like `worker.run_scene` (without plots) and keeps every dot's error.
`--ref` adds an older run of the same scenes (e.g. the first 33x33 calibration) as grey reference markers.
Writes per_dot_errors.npz, summary.json/.csv and flatwall_compare.png/.pdf into <runs>.
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import approaches  # noqa: E402
import calibrate  # noqa: E402
import sl_tilt_correction  # noqa: E402
from dot_calibration import DotCalibration  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ILLUMINATIONS = {"10x10": ("10x10", "#0072B2"), "Coherent": ("33x33", "#D55E00")}
APPROACHES = (1, 2, 3)
GROSS_MM = 50.0        # [mm] a dot is a gross error when |e − median| exceeds this
LEAK_TOL_M = 0.001     # [m] scene within this of a calibration distance → not an independent test


def safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def run_illumination(run_dir):
    """{(scene, approach): errors [mm]} plus scene GTs and calibration distances for one GUI run folder."""
    job = json.loads((run_dir / "job.json").read_text())["test"]
    overrides = run_dir / "overrides.json"
    if overrides.exists():
        for k, v in json.loads(overrides.read_text()).get("approaches", {}).items():
            setattr(approaches, k, v)
    dotCal = DotCalibration()
    cal = calibrate.load_saved_calibration(dotCal, name=str(Path(job["calibration"]).parent),
                                           light_path_correction=job.get("light_path_correction", False))
    cal = sl_tilt_correction.correct_cal(cal, **job.get("tilt", {}))
    errors, gts = {}, {}
    for scene in job["scenes"]:
        tof = run_dir / safe(scene["name"]) / "tof" / f"{safe(scene['name'])}.pcd"
        if not tof.exists():
            print(f"  {run_dir.name}/{scene['name']}: no PCD, skipped")
            continue
        gts[scene["name"]] = scene["gt"]
        approaches.GT_DISTANCE = scene["gt"]
        with contextlib.redirect_stdout(io.StringIO()):
            setup = approaches.common_setup(dotCal, cal, scene.get("sl"), str(tof))
            for n in APPROACHES:
                z = np.asarray(approaches.RUNNERS[n](dotCal, cal, setup, make_plots=False)["depths"], float)
                errors[(scene["name"], n)] = np.where(np.isfinite(z) & (z > 1e-6), (z - scene["gt"]) * 1e3, np.nan)
        plt.close("all")
        print(f"  {run_dir.name}/{scene['name']} done")
    return errors, gts, list(cal.get("cal_dists", [])), job["calibration"]


def stats(e):
    v = e[np.isfinite(e)]
    if v.size == 0:
        return dict(n_valid=0, n_total=int(e.size))
    med = float(np.median(v))
    inl = v[np.abs(v - med) <= GROSS_MM]
    return dict(n_valid=int(v.size), n_total=int(e.size), valid_pct=100 * v.size / e.size,
                mean=float(v.mean()), median=med, sigma_mad=float(1.4826 * np.median(np.abs(v - med))),
                gross_pct=100 * (1 - inl.size / v.size), mean_inl=float(inl.mean()) if inl.size else np.nan,
                std_inl=float(inl.std()) if inl.size else np.nan,
                p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "Results" / "flatwall_6.11_v2"))
    p.add_argument("--ref", default=str(ROOT / "Results" / "flatwall_6.11"),
                   help="older run of the same scenes, shown as grey markers (only the 33x33 folder); '' for none")
    args = p.parse_args()
    runs = Path(args.runs)

    data = {}  # label → (errors, gts, cal_dists)
    for folder, (label, _) in ILLUMINATIONS.items():
        print(f"{label}: {runs / folder}")
        errs, gts, cd, calp = run_illumination(runs / folder)
        data[label] = (errs, gts, cd)
        print(f"  calibration {Path(calp).parent.name}")
    ref = None
    if args.ref and (Path(args.ref) / "Coherent" / "job.json").exists():
        print(f"reference 33x33: {Path(args.ref) / 'Coherent'}")
        errs, gts, cd, calp = run_illumination(Path(args.ref) / "Coherent")
        ref = (errs, gts, Path(calp).parent.name)

    # ── tables ──
    rows = []
    for label, (errs, gts, cd) in list(data.items()) + ([("33x33 old", ref[:2] + ([],))] if ref else []):
        for (scene, n), e in errs.items():
            leak = any(abs(gts[scene] - c) < LEAK_TOL_M for c in cd)
            rows.append(dict(illumination=label, scene=scene, gt_m=gts[scene], approach=n,
                             in_calibration=leak, **stats(e)))
    (runs / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(runs / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(runs / "per_dot_errors.npz",
                        **{f"{label}|{scene}|A{n}": e for label, (errs, _, _) in data.items()
                           for (scene, n), e in errs.items()},
                        **({f"33x33 old|{scene}|A{n}": e for (scene, n), e in ref[0].items()} if ref else {}))

    print(f"\n{'illum':10s} {'scene':22s} {'GT':>6s} A | {'n':>9s} {'valid%':>6s} {'median':>8s} {'σ_MAD':>7s}"
          f" {'gross%':>6s} {'mean_inl':>8s} {'std_inl':>7s}   [mm]")
    for r in sorted(rows, key=lambda r: (r["approach"], r["gt_m"], r["illumination"])):
        if r["n_valid"] == 0:
            continue
        print(f"{r['illumination']:10s} {r['scene']:22s} {r['gt_m']:6.3f} {r['approach']} | "
              f"{r['n_valid']:4d}/{r['n_total']:<4d} {r['valid_pct']:6.1f} {r['median']:8.1f} {r['sigma_mad']:7.1f}"
              f" {r['gross_pct']:6.1f} {r['mean_inl']:8.1f} {r['std_inl']:7.1f}"
              + ("   (scene in calibration)" if r["in_calibration"] else ""))

    # ── figure ──
    plt.rcParams.update({"font.size": 9})
    fig, axs = plt.subplots(3, 3, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    all_gt = sorted({g for _, (_, gts, _) in data.items() for g in gts.values()})
    width = 0.035 * (max(all_gt) - min(all_gt) + 0.5)
    for j, n in enumerate(APPROACHES):
        ax0, ax1, ax2 = axs[:, j]
        for k, (folder, (label, color)) in enumerate(ILLUMINATIONS.items()):
            errs, gts, cd = data[label]
            off = (k - 0.5) * 1.15 * width
            scenes = sorted(gts, key=gts.get)
            pos, vals, st = [], [], []
            for s in scenes:
                e = errs[(s, n)]
                v = e[np.isfinite(e)]
                if v.size:
                    pos.append(gts[s] + off)
                    vals.append(v)
                    st.append((gts[s], stats(e), any(abs(gts[s] - c) < LEAK_TOL_M for c in cd)))
            ax0.boxplot(vals, positions=pos, widths=width, whis=(5, 95), showfliers=True, manage_ticks=False,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
                        medianprops=dict(color=color, lw=2), whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        flierprops=dict(marker=".", ms=2, mec=color, mfc=color, alpha=0.4))
            for (g, s, leak), x in zip(st, pos):
                if leak:
                    ax0.annotate("cal", (x, s["p95"]), xytext=(0, 6), textcoords="offset points",
                                 ha="center", fontsize=7, color=color)
            g = [t[0] for t in st]
            ax1.plot(g, [t[1]["sigma_mad"] for t in st], "o-", color=color, label=label)
            ax2.plot(g, [t[1]["valid_pct"] for t in st], "o-", color=color, label=f"{label} valid")
            ax2.plot(g, [t[1]["gross_pct"] for t in st], "s--", color=color, mfc="none",
                     label=f"{label} gross (>{GROSS_MM:.0f} mm)")
        if ref:
            rerrs, rgts, rname = ref
            g = sorted(rgts.values())
            sc = sorted(rgts, key=rgts.get)
            med = [stats(rerrs[(s, n)]).get("median", np.nan) for s in sc]
            ax0.plot(np.array(g) + 0.5 * 1.15 * width, med, "x", color="0.4", ms=8, mew=2,
                     label="33x33 old cal (median)")
            ax1.plot(g, [stats(rerrs[(s, n)]).get("sigma_mad", np.nan) for s in sc], "x:", color="0.5",
                     label="33x33 old cal")
        ax0.axhline(0, color="k", lw=0.8)
        ax0.set_yscale("symlog", linthresh=20)
        ax0.set_title(f"Approach {n}", fontsize=11)
        ax0.grid(alpha=0.3, which="both")
        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 105)
        ax2.set_xlabel("GT distance [m]")
        if j == 0:
            ax0.set_ylabel("error = Z − GT [mm]\n(box IQR, whiskers 5–95 %, symlog beyond ±20)")
            ax1.set_ylabel("σ_MAD [mm]")
            ax2.set_ylabel("share of trails [%]")
    h0 = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.35, ec=c) for _, c in ILLUMINATIONS.values()]
    l0 = [lab for lab, _ in ILLUMINATIONS.values()]
    if ref:
        h0.append(plt.Line2D([], [], ls="", marker="x", color="0.4", mew=2))
        l0.append("33x33 old cal, median")
    axs[0, 0].legend(h0, l0, fontsize=8, loc="lower right")
    axs[1, 2].legend(fontsize=8)
    axs[2, 2].legend(fontsize=7, ncol=2, loc="center right")
    fig.suptitle("Flat wall 6.11: 10x10 vs. 33x33 (v2 calibration)  —  'cal' = scene distance is a calibration frame",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(runs / f"flatwall_compare.{ext}", dpi=150)
    print(f"\nwrote {runs / 'flatwall_compare.png'}")


if __name__ == "__main__":
    main()
