"""
SL tilt correction — small projector rotations after the dot calibration.

A dot trail is affine in w = 1/z:  p(w) = p_inf + w·g  (dot_calibration.build_dot_trails_1d).
A small rotation δ of the projector moves every p_inf by ≈ f·δ pixels and leaves g unchanged, so along
a trail the same image position now belongs to a different depth: w_true = w_cal + f·δ / gain, with
gain = |g| [px per 1/m] of that trail. The correction therefore keeps the trail pixels (u, v) and only
shifts their depth labels w — one parameter for the whole pattern.

  tilt_from_offset   manual input: "the SL reads `offset_mm` too far at `z_ref_m`" → δ
  apply_tilt_trails  shift the trails' w by δ (apply_tilt: same on a calibration dict)
  sl_dot_peaks       SL-only dot positions (no ToF): one brightness peak with a 2-D dot per trail
  estimate_tilt      δ from frames with a known depth per pixel ray (wall at a known distance, scene GT)

CLI:  python sl_tilt_correction.py --cal <folder> (--offset-mm 90 --z-ref 1.1 | --tilt-mrad X) --name <out>
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import approaches


def focal(K, mode):
    # Focal length along the travel axis of the trails (as approaches.sigma_models).
    K = np.asarray(K, float)
    return abs(float(K[0, 0])) if mode == "x" else abs(float(K[1, 1]))


def trail_gain(trail):
    # Pixel travel per unit w of one trail (as approaches.ab_trail_candidates).
    w_span = float(trail["w"][0] - trail["w"][-1])
    return float(np.hypot(trail["u"][0] - trail["u"][-1], trail["v"][0] - trail["v"][-1]) / max(w_span, 1e-9))


def tilt_from_offset(trails, K, mode, offset_mm, z_ref_m):
    """Tilt δ [mrad] for "the SL depth is `offset_mm` too far at the true distance `z_ref_m`"
    (negative offset: too close). Returns dict(tilt_mrad, dpx, dw)."""
    dw = 1.0 / z_ref_m - 1.0 / (z_ref_m + offset_mm * 1e-3)
    gain = float(np.median([trail_gain(t) for t in trails if t is not None]))
    dpx = dw * gain
    return dict(tilt_mrad=dpx / focal(K, mode) * 1e3, dpx=dpx, dw=dw)


def apply_tilt_trails(trails, K, mode, tilt_mrad):
    """Copy of `trails` with every depth label shifted by the tilt: w += f·δ / gain_i.
    Samples that would end up beyond infinity (w ≤ 0, negative tilt) are dropped."""
    dpx = focal(K, mode) * tilt_mrad * 1e-3
    out = []
    for t in trails:
        if t is None:
            out.append(None)
            continue
        w = np.asarray(t["w"], float) + dpx / trail_gain(t)
        keep = w > 1e-9
        if keep.sum() < 2:
            out.append(None)
            continue
        nt = {"u": np.asarray(t["u"], float)[keep], "v": np.asarray(t["v"], float)[keep], "w": w[keep]}
        nt["z"] = 1.0 / nt["w"]
        out.append(nt)
    return out


def apply_tilt(cal, tilt_mrad, source="manual"):
    # Calibration dict with tilted trails; the shift is recorded in its metadata.
    mode = cal.get("mode", cal.get("metadata", {}).get("mode", "x"))
    new = copy.copy(cal)
    new["trails"] = apply_tilt_trails(cal["trails"], cal["K"], mode, tilt_mrad)
    new["metadata"] = dict(cal.get("metadata", {}), sl_tilt_mrad=float(tilt_mrad),
                           sl_tilt_px=float(focal(cal["K"], mode) * tilt_mrad * 1e-3), sl_tilt_source=source)
    return new


def sl_dot_peaks(dotCal, setup, w_ref=None):
    """SL-only dot positions of one frame (the ToF depth is not used).

    Per trail the brightness peaks with a 2-D dot (blob) within BLOB_RADIUS; of these the most
    prominent one, or — with `w_ref(u, v)` (reference 1/z at the peak pixels) — the one closest to the
    reference. Refined on the trail as in approaches.ab_trail_candidates.
    Returns a dict of arrays: i (trail index), u, v, w, gain, d_perp (blob centre ↔ trail [px])."""
    ap = approaches
    sl = setup["sl_gray"]
    rows = []
    for i, tr in enumerate(setup["trails"]):
        if tr is None or setup["blob_tree"] is None:
            continue
        ab = ap.saturation_correct(ap.sample_trail_brightness(sl, tr), setup["sat_level"])
        if not np.any(np.isfinite(ab)):
            continue
        sig_s, base, pk = ap.trail_peaks(np.where(np.isfinite(ab), ab, 0.0))
        if len(pk):
            d_blob, _ = setup["blob_tree"].query(np.c_[tr["u"][pk], tr["v"][pk]])
            pk = pk[d_blob <= ap.BLOB_RADIUS]
        if not len(pk):
            continue
        if w_ref is None:
            k0 = int(pk[np.argmax((sig_s - base)[pk])])
        else:
            wr = np.asarray(w_ref(tr["u"][pk], tr["v"][pk]), float)
            if not np.any(np.isfinite(wr)):
                continue
            k0 = int(pk[np.nanargmin(np.abs(tr["w"][pk] - wr))])
        k_sub, d_perp = ap.refine_on_trail(dotCal, sl, tr, ap.quad_subpixel(sig_s, k0, baseline=base),
                                           k_anchor=k0)
        ks = np.arange(len(tr["u"]), dtype=float)
        rows.append((i, np.interp(k_sub, ks, tr["u"]), np.interp(k_sub, ks, tr["v"]),
                     np.interp(k_sub, ks, tr["w"]), trail_gain(tr), d_perp))
    keys = ("i", "u", "v", "w", "gain", "d_perp")
    if not rows:
        return {k: np.zeros(0) for k in keys}
    arr = np.array(rows, float)
    out = {k: arr[:, j] for j, k in enumerate(keys)}
    out["i"] = out["i"].astype(int)
    return out


def estimate_tilt(dotCal, setups_and_refs, K, mode, max_dw=0.5, iters=3):
    """Tilt δ from frames with a known depth: `setups_and_refs` = [(setup, z_known(u, v) → axial z [m])].

    Per dot the peak closest to the reference (shifted by the current estimate, so a large tilt does
    not pick the neighbouring dot) gives dpx_i = (w_known − w_peak)·gain_i; δ is their median."""
    f = focal(K, mode)
    dpx_med = 0.0
    for _ in range(iters):
        all_dpx, all_perp = [], []
        for setup, z_known in setups_and_refs:
            w_known = lambda u, v, zk=z_known: 1.0 / np.asarray(zk(np.asarray(u, float), np.asarray(v, float)), float)
            pk = sl_dot_peaks(dotCal, setup,
                              w_ref=lambda u, v, wk=w_known: wk(u, v) - dpx_med / np.median(
                                  [trail_gain(t) for t in setup["trails"] if t is not None]))
            if not len(pk["w"]):
                continue
            dw = w_known(pk["u"], pk["v"]) - pk["w"]
            ok = np.isfinite(dw) & (np.abs(dw) < max_dw)
            all_dpx.append(dw[ok] * pk["gain"][ok])
            all_perp.append(pk["d_perp"][ok])
        dpx = np.concatenate(all_dpx) if all_dpx else np.zeros(0)
        if not len(dpx):
            return None
        dpx_med = float(np.median(dpx))
    perp = np.concatenate(all_perp)
    return dict(tilt_mrad=dpx_med / f * 1e3, dpx=dpx_med,
                dpx_mad=float(1.4826 * np.median(np.abs(dpx - dpx_med))), n=int(len(dpx)),
                d_perp_med=float(np.nanmedian(perp)) if np.any(np.isfinite(perp)) else np.nan)


def describe(tilt_mrad, K, mode, trails, z=(0.5, 1.0, 1.5)):
    # One line: δ, pixel shift and the depth change it causes at a few distances.
    dpx = focal(K, mode) * tilt_mrad * 1e-3
    gain = float(np.median([trail_gain(t) for t in trails if t is not None]))
    dz = ", ".join(f"{zz:.1f} m → {(1.0 / (1.0 / zz + dpx / gain) - zz) * 1e3:+.0f} mm" for zz in z)
    return (f"SL tilt δ = {tilt_mrad:+.3f} mrad ({np.degrees(tilt_mrad * 1e-3):+.4f}°), "
            f"Δp = {dpx:+.2f} px along the trails; depth change: {dz}")


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cal", required=True, help="calibration folder in Calibrations/")
    ap_.add_argument("--offset-mm", type=float, help="SL reads this much too far (+) / too close (−) …")
    ap_.add_argument("--z-ref", type=float, help="… at this true distance [m]")
    ap_.add_argument("--tilt-mrad", type=float, help="tilt directly")
    ap_.add_argument("--name", required=True, help="output calibration folder")
    a = ap_.parse_args()
    root = Path(__file__).resolve().parent.parent / "Calibrations"
    raw = json.loads((root / a.cal / "calibration.json").read_text())
    cal = approaches.load_calibration(str(root / a.cal / "calibration.json"))
    mode = cal.get("mode", cal.get("metadata", {}).get("mode", "x"))
    if a.tilt_mrad is not None:
        tilt = a.tilt_mrad
    elif a.offset_mm is not None and a.z_ref is not None:
        tilt = tilt_from_offset(cal["trails"], cal["K"], mode, a.offset_mm, a.z_ref)["tilt_mrad"]
    else:
        ap_.error("give --tilt-mrad or --offset-mm with --z-ref")
    trails = apply_tilt_trails(cal["trails"], cal["K"], mode, tilt)
    raw["trails"] = [None if t is None else {"u": t["u"].tolist(), "v": t["v"].tolist(), "w": t["w"].tolist()}
                     for t in trails]
    raw["metadata"] = dict(raw.get("metadata", {}), sl_tilt_mrad=tilt, sl_tilt_source=f"from {a.cal}",
                           sl_tilt_px=focal(cal["K"], mode) * tilt * 1e-3)
    out = root / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(raw))
    print(describe(tilt, cal["K"], mode, cal["trails"]))
    print(f"→ {out / 'calibration.json'}")


if __name__ == "__main__":
    main()
