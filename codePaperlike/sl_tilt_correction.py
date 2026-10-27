#!/usr/bin/env python3
"""SL tilt correction: a small projector rotation δ after calibration shifts every trail's depth labels by w += f·δ / gain.

Usage:
    python sl_tilt_correction.py --cal <folder> (--offset-mm 90 --z-ref 1.1 | --tilt-mrad X) --name <out>
"""

import argparse
import copy

import numpy as np

import approaches
import calibrate
from approaches import trail_gain


def focal(K, mode):
    """Focal length along the trail travel axis."""
    K = np.asarray(K, float)
    return abs(float(K[0, 0])) if mode == "x" else abs(float(K[1, 1]))


def cal_mode(cal):
    """Travel mode of a calibration dict."""
    return cal.get("mode") or cal.get("metadata", {}).get("mode", "x")


def tilt_from_offset(trails, K, mode, offset_mm, z_ref_m):
    """Tilt for 'the SL reads offset_mm too far at z_ref_m'; returns dict(tilt_mrad, dpx, dw)."""
    dw = 1.0 / z_ref_m - 1.0 / (z_ref_m + offset_mm * 1e-3)
    dpx = dw * float(np.median([trail_gain(t) for t in trails if t is not None]))
    return dict(tilt_mrad=dpx / focal(K, mode) * 1e3, dpx=dpx, dw=dw)


def apply_tilt_trails(trails, K, mode, tilt_mrad):
    """Copy of trails with depth labels shifted by the tilt; samples beyond infinity are dropped."""
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
        out.append({"u": np.asarray(t["u"], float)[keep], "v": np.asarray(t["v"], float)[keep],
                    "w": w[keep], "z": 1.0 / w[keep]})
    return out


def apply_tilt(cal, tilt_mrad, source="manual"):
    """Copy of a calibration dict with tilted trails, the tilt recorded in its metadata."""
    mode = cal_mode(cal)
    new = copy.copy(cal)
    new["trails"] = apply_tilt_trails(cal["trails"], cal["K"], mode, tilt_mrad)
    new["metadata"] = dict(cal.get("metadata", {}), sl_tilt_mrad=float(tilt_mrad),
                           sl_tilt_px=float(focal(cal["K"], mode) * tilt_mrad * 1e-3), sl_tilt_source=source)
    return new


def apply_z_shift(cal, z_shift_mm):
    """Copy of a calibration dict whose trail depth labels are z − z_shift_mm (calibration distances measured too far)."""
    dz = z_shift_mm * 1e-3
    trails = []
    for t in cal["trails"]:
        if t is None:
            trails.append(None)
            continue
        z = np.asarray(t["z"], float) - dz
        keep = z > 1e-6
        if keep.sum() < 2:
            trails.append(None)
            continue
        trails.append({"u": np.asarray(t["u"], float)[keep], "v": np.asarray(t["v"], float)[keep],
                       "w": 1.0 / z[keep], "z": z[keep]})
    new = copy.copy(cal)
    new["trails"] = trails
    new["metadata"] = dict(cal.get("metadata", {}), sl_z_shift_mm=float(z_shift_mm))
    return new


def correct_cal(cal, offset_mm=None, z_ref=None, tilt_mrad=None, z_shift_mm=None):
    """Calibration with tilt correction from a direct tilt or an observed offset, and/or a constant z shift of
    the trail labels (z_shift_mm > 0: the SL reads that much too far at every distance); unchanged if none is given."""
    if z_shift_mm:
        print(f"SL z shift: trail depth labels − {z_shift_mm:.1f} mm")
        cal = apply_z_shift(cal, z_shift_mm)
    if tilt_mrad is not None and offset_mm is not None:
        raise ValueError("give either offset_mm + z_ref or tilt_mrad, not both (set the other to None)")
    if tilt_mrad is None and offset_mm is not None:
        if z_ref is None:
            raise ValueError("offset_mm needs z_ref: the distance [m] at which the SL reads offset_mm too far")
        tilt_mrad = tilt_from_offset(cal["trails"], cal["K"], cal_mode(cal), offset_mm, z_ref)["tilt_mrad"]
    if tilt_mrad is None:
        print("SL tilt correction off — trails as calibrated")
        return cal
    print(describe(tilt_mrad, cal["K"], cal_mode(cal), cal["trails"]))
    return apply_tilt(cal, tilt_mrad)


def sl_dot_peaks(dotCal, setup, w_ref=None):
    """SL-only refined dot position per trail (strongest blob peak, or the one closest to w_ref(u, v))."""
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
            pk = pk[setup["blob_tree"].query(np.c_[tr["u"][pk], tr["v"][pk]])[0] <= ap.BLOB_RADIUS]
        if not len(pk):
            continue
        if w_ref is None:
            k0 = int(pk[np.argmax((sig_s - base)[pk])])
        else:
            wr = np.asarray(w_ref(tr["u"][pk], tr["v"][pk]), float)
            if not np.any(np.isfinite(wr)):
                continue
            k0 = int(pk[np.nanargmin(np.abs(tr["w"][pk] - wr))])
        k_sub, d_perp = ap.refine_on_trail(dotCal, sl, tr, ap.quad_subpixel(sig_s, k0, baseline=base), k_anchor=k0)
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
    """Tilt from frames with known depth [(setup, z_known(u, v))]: median pixel shift of the SL peaks."""
    dpx_med = 0.0
    for _ in range(iters):
        all_dpx, all_perp = [], []
        for setup, z_known in setups_and_refs:
            def w_known(u, v, zk=z_known):
                return 1.0 / np.asarray(zk(np.asarray(u, float), np.asarray(v, float)), float)
            gain = np.median([trail_gain(t) for t in setup["trails"] if t is not None])
            pk = sl_dot_peaks(dotCal, setup, w_ref=lambda u, v, wk=w_known: wk(u, v) - dpx_med / gain)
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
    return dict(tilt_mrad=dpx_med / focal(K, mode) * 1e3, dpx=dpx_med,
                dpx_mad=float(1.4826 * np.median(np.abs(dpx - dpx_med))), n=int(len(dpx)),
                d_perp_med=float(np.nanmedian(perp)) if np.any(np.isfinite(perp)) else np.nan)


def describe(tilt_mrad, K, mode, trails, z=(0.5, 1.0, 1.5)):
    """One line: tilt, pixel shift and the resulting depth change at a few distances."""
    dpx = focal(K, mode) * tilt_mrad * 1e-3
    gain = float(np.median([trail_gain(t) for t in trails if t is not None]))
    dz = ", ".join(f"{zz:.1f} m → {(1.0 / (1.0 / zz + dpx / gain) - zz) * 1e3:+.0f} mm" for zz in z)
    return (f"SL tilt δ = {tilt_mrad:+.3f} mrad ({np.degrees(tilt_mrad * 1e-3):+.4f}°), "
            f"Δp = {dpx:+.2f} px along the trails; depth change: {dz}")


def main():
    """Command-line entry: write a tilt-corrected copy of a calibration."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cal", required=True, help="calibration folder in Calibrations/")
    p.add_argument("--offset-mm", type=float, help="SL reads this much too far (+) / too close (−) …")
    p.add_argument("--z-ref", type=float, help="… at this true distance [m]")
    p.add_argument("--tilt-mrad", type=float, help="tilt directly")
    p.add_argument("--name", required=True, help="output calibration folder")
    a = p.parse_args()
    if a.tilt_mrad is None and (a.offset_mm is None or a.z_ref is None):
        p.error("give --tilt-mrad or --offset-mm with --z-ref")

    cal = approaches.load_calibration(str(calibrate.CAL_ROOT / a.cal / "calibration.json"))
    tilt = a.tilt_mrad
    if tilt is None:
        tilt = tilt_from_offset(cal["trails"], cal["K"], cal_mode(cal), a.offset_mm, a.z_ref)["tilt_mrad"]
    print(describe(tilt, cal["K"], cal_mode(cal), cal["trails"]))
    calibrate.save_calibration(apply_tilt(cal, tilt, source=f"from {a.cal}"), a.name)


if __name__ == "__main__":
    main()
