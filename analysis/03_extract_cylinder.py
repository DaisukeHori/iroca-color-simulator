#!/usr/bin/env python3
# =========================================================
# 03_extract_cylinder.py — 棒巻き毛束の角度別 視覚Lab 抽出
# =========================================================
# 使い方:
#   python 03_extract_cylinder.py             # 全件処理
#   python 03_extract_cylinder.py a05         # 特定サンプルのみ
#
# 幾何モデル:
#   毛束を巻いた棒(半径R)が水平に写っている。
#   画像Y座標 y と円筒法線角θの関係:  y = axis_y + R·sin(θ)
#   θ=0 は apex(法線がカメラ正対)。±両側の同角度帯を平均する。
#
# 処理内容:
#   1. リニアTIFF読み込み
#   2. 棒領域の指定 (config.yaml manual_rods / 自動推定+確認画像)
#   3. 各角度帯 (0/15/30/45/60°) の帯状ROIからロバスト抽出
#      - 輝度上位パーセンタイルをスペキュラとして除外
#      - median で代表RGB決定
#   4. 該当画像の CCM (02の出力) を適用 → XYZ → Lab(D50)
#   5. output/visual_labs.csv に追記 + 確認画像出力
# =========================================================

import sys
import os
import glob
import json
import csv
import yaml
import numpy as np
import cv2
import tifffile
import colour


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 02_detect_chart.py と同じRPCC適用関数 (依存を切るため再掲)
from colour.characterisation import polynomial_expansion_Finlayson2015

def apply_ccm_rpcc(rgb_linear: np.ndarray, M: np.ndarray, degree: int = 2) -> np.ndarray:
    single = rgb_linear.ndim == 1
    rgb = np.atleast_2d(rgb_linear)
    expanded = np.array([
        polynomial_expansion_Finlayson2015(v, degree=degree, root_polynomial_expansion=True)
        for v in rgb
    ])
    XYZ = expanded @ np.asarray(M).T
    return XYZ[0] if single else XYZ


def auto_detect_rod(img_lin: np.ndarray):
    """棒巻き毛束領域の自動推定。
    前提: 毛束は画像中央付近の水平帯で、背景(明るめ)より暗い。
    戻り値: dict(axis_y, top_y, bottom_y, x_start, x_end) or None
    """
    h, w = img_lin.shape[:2]
    # 輝度 (リニアY近似)
    lum = img_lin @ np.array([0.2126, 0.7152, 0.0722])
    # 中央60%の列で行平均
    x0, x1 = int(w * 0.2), int(w * 0.8)
    row_mean = lum[:, x0:x1].mean(axis=1)
    # 大津的に: 暗い帯 = row_mean が全体中央値の60%未満の連続区間
    thresh = np.median(row_mean) * 0.6
    dark = row_mean < thresh
    # 最長の暗い連続区間を探す
    best_len, best_start = 0, -1
    cur_len, cur_start = 0, -1
    for i, v in enumerate(dark):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    if best_len < h * 0.03:   # 高さ3%未満の帯は棒とみなさない
        return None
    top_y = best_start
    bottom_y = best_start + best_len - 1
    axis_y = (top_y + bottom_y) // 2
    return {
        "axis_y": int(axis_y),
        "top_y": int(top_y),
        "bottom_y": int(bottom_y),
        "x_start": int(x0),
        "x_end": int(x1),
    }


def extract_angles(img_lin: np.ndarray, rod: dict, cfg: dict):
    """角度帯ごとのロバストRGB (リニア) を抽出。
    戻り値: {angle_deg: {"rgb": [r,g,b], "n_pixels": int, "specular_cut": int}}
    """
    cyl = cfg["cylinder"]
    angles = cyl["angles_deg"]
    half_bw = np.deg2rad(cyl["band_halfwidth_deg"])
    cut_pct = cyl["specular_percentile_cut"]
    min_px = cyl["min_pixels_per_band"]

    axis_y = rod["axis_y"]
    R = (rod["bottom_y"] - rod["top_y"]) / 2.0
    x0, x1 = rod["x_start"], rod["x_end"]

    lum_w = np.array([0.2126, 0.7152, 0.0722])
    results = {}
    for a in angles:
        th = np.deg2rad(a)
        collected = []
        # ±両側 (上側=−θ, 下側=+θ)。a=0 は1帯のみ。
        sides = [th] if a == 0 else [th, -th]
        for s in sides:
            y_lo = axis_y + R * np.sin(s - half_bw)
            y_hi = axis_y + R * np.sin(s + half_bw)
            ylo, yhi = int(round(min(y_lo, y_hi))), int(round(max(y_lo, y_hi)))
            ylo = max(rod["top_y"], ylo)
            yhi = min(rod["bottom_y"], yhi)
            if yhi <= ylo:
                continue
            roi = img_lin[ylo:yhi + 1, x0:x1].reshape(-1, 3)
            collected.append(roi)
        if not collected:
            results[a] = None
            continue
        px = np.concatenate(collected, axis=0)

        # スペキュラ除去: 輝度上位 (100-cut_pct)% をカット
        lum = px @ lum_w
        thr = np.percentile(lum, cut_pct)
        keep = px[lum <= thr]
        n_cut = int(len(px) - len(keep))
        if len(keep) < min_px:
            results[a] = {"rgb": None, "n_pixels": int(len(keep)), "specular_cut": n_cut,
                          "warning": f"帯内ピクセル{len(keep)} < 最低{min_px}"}
            continue
        rgb = np.median(keep, axis=0)
        results[a] = {"rgb": rgb.tolist(), "n_pixels": int(len(keep)), "specular_cut": n_cut}
    return results


def main():
    cfg = load_config()
    tiff_dir = cfg["linear_tiff_dir"]
    chk_dir = cfg["check_image_dir"]
    ccm_dir = os.path.join(cfg["output_dir"], "ccm")
    os.makedirs(chk_dir, exist_ok=True)

    degree = int(cfg["ccm"].get("degree", 2))
    manual_rods = cfg["cylinder"].get("manual_rods") or {}
    illum_name = cfg["colorimetry"]["illuminant"]
    # D50 白色点 (xy)
    illum_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][illum_name]

    if len(sys.argv) > 1:
        stems = [os.path.splitext(a)[0] for a in sys.argv[1:]]
        targets = [os.path.join(tiff_dir, s + ".tiff") for s in stems]
    else:
        targets = sorted(glob.glob(os.path.join(tiff_dir, "*.tiff")))

    if not targets:
        print(f"リニアTIFFが見つかりません: {tiff_dir}")
        sys.exit(1)

    csv_path = os.path.join(cfg["output_dir"], "visual_labs.csv")
    rows = []
    print(f"=== 円筒解析: {len(targets)}ファイル ===")

    for tp in targets:
        base = os.path.splitext(os.path.basename(tp))[0]
        try:
            # --- CCM読み込み ---
            ccm_path = os.path.join(ccm_dir, base + "_ccm.json")
            if not os.path.exists(ccm_path):
                print(f"  ⏭️  {base}: CCMなし (02_detect_chart.py 未実行 or 失敗) → スキップ")
                continue
            with open(ccm_path, "r", encoding="utf-8") as f:
                ccm = json.load(f)
            M = np.array(ccm["M"])
            white_norm = ccm["white_norm"]

            img16 = tifffile.imread(tp)
            img_lin = img16.astype(np.float64) / 65535.0

            # --- 棒領域 ---
            rod = None
            for k in [base, base + ".RW2", base + ".rw2"]:
                if k in manual_rods:
                    rod = manual_rods[k]
                    print(f"  {base}: manual_rods 使用")
                    break
            if rod is None:
                rod = auto_detect_rod(img_lin)
                if rod is None:
                    print(f"  ❌ {base}: 棒の自動検出失敗 → config.yaml manual_rods に座標記入して再実行")
                    continue

            # --- 角度別抽出 ---
            res = extract_angles(img_lin, rod, cfg)

            # --- CCM適用 → Lab (D50) ---
            row = {"sample": base}
            for a, r in res.items():
                if r is None or r.get("rgb") is None:
                    row[f"L_{a}"] = row[f"a_{a}"] = row[f"b_{a}"] = ""
                    if r:
                        print(f"    ⚠️ {base} {a}°: {r.get('warning','')}")
                    continue
                rgb = np.array(r["rgb"]) / white_norm * 0.9   # CCM学習時と同じ正規化
                XYZ = apply_ccm_rpcc(rgb, M, degree=degree)
                Lab = colour.XYZ_to_Lab(XYZ, illuminant=illum_xy)
                row[f"L_{a}"] = round(float(Lab[0]), 2)
                row[f"a_{a}"] = round(float(Lab[1]), 2)
                row[f"b_{a}"] = round(float(Lab[2]), 2)
                row[f"npx_{a}"] = r["n_pixels"]
            rows.append(row)

            # --- 確認画像 ---
            img8 = (np.clip(img_lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
            vis = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
            axis_y, R = rod["axis_y"], (rod["bottom_y"] - rod["top_y"]) / 2.0
            x0, x1 = rod["x_start"], rod["x_end"]
            cv2.rectangle(vis, (x0, rod["top_y"]), (x1, rod["bottom_y"]), (255, 200, 0), 2)
            cv2.line(vis, (x0, axis_y), (x1, axis_y), (0, 255, 0), 2)
            for a in cfg["cylinder"]["angles_deg"]:
                for s in ([1] if a == 0 else [1, -1]):
                    y = int(round(axis_y + R * np.sin(np.deg2rad(a * s))))
                    cv2.line(vis, (x0, y), (x1, y), (0, 0, 255), 1)
                    cv2.putText(vis, f"{a}", (x1 + 8, y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            h, w = vis.shape[:2]
            scale = 2000 / max(h, w)
            if scale < 1:
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(chk_dir, base + "_cylinder.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            L_vals = " / ".join(f"{a}°:{row.get('L_'+str(a),'—')}" for a in cfg["cylinder"]["angles_deg"])
            print(f"  ✅ {base}: L* = {L_vals}")

        except Exception as e:
            print(f"  ❌ {base}: {e}")

    # --- CSV出力 ---
    if rows:
        angles = cfg["cylinder"]["angles_deg"]
        fields = ["sample"]
        for a in angles:
            fields += [f"L_{a}", f"a_{a}", f"b_{a}", f"npx_{a}"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"=== 完了: {len(rows)}件 → {csv_path} / 確認画像: {chk_dir}/*_cylinder.jpg ===")
    else:
        print("=== 出力なし ===")


if __name__ == "__main__":
    main()
