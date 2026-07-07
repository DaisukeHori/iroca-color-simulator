#!/usr/bin/env python3
# =========================================================
# 02_detect_chart.py — ColorChecker検出 + CCM計算 (RPCC)
# =========================================================
# 使い方:
#   python 02_detect_chart.py                 # linear_tiff 全件処理
#   python 02_detect_chart.py a05             # 特定サンプルのみ
#
# 処理内容:
#   1. リニアTIFFから OpenCV mcc で ColorChecker Classic を自動検出
#      (失敗時は config.yaml の manual_corners を参照)
#   2. 24パッチの実測リニアRGBを抽出
#   3. colour-science で Root-Polynomial CCM (Finlayson 2015) を計算
#   4. 24パッチの残差ΔE00 をレポート
#   5. CCM を output/ccm/{basename}_ccm.json に保存
#   6. 確認画像 (パッチ位置+補正前後スウォッチ) を出力
# =========================================================

import sys
import os
import glob
import json
import yaml
import numpy as np
import cv2
import tifffile
import colour
from colour.characterisation import matrix_augmented_Cheung2004, polynomial_expansion_Finlayson2015
from colour.difference import delta_E_CIE2000


# --- ColorChecker Classic (After Nov 2014) の公称値 ---
# colour-science 内蔵の ColorChecker24 - After November 2014 データを使用
def get_reference_lab_d50():
    """ColorChecker 24パッチの公称Lab値 (D50) を取得。
    パッチ順序: 左上(dark skin)→右へ、行順で24番(black)まで。
    """
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    # cc.data は {name: xyY} (illuminant = cc.illuminant, 通常 ICC D50)
    names = list(cc.data.keys())
    xyY = np.array([cc.data[n] for n in names])
    XYZ = colour.xyY_to_XYZ(xyY)
    # cc.illuminant は xy 座標
    Lab = colour.XYZ_to_Lab(XYZ, illuminant=cc.illuminant)
    return names, Lab, XYZ, cc.illuminant


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_chart_mcc(img8: np.ndarray):
    """OpenCV mcc で ColorChecker を検出し、24パッチ矩形の中心座標リストを返す。
    戻り値: (patch_centers [24x2], chart_box [4x2]) or (None, None)
    """
    detector = cv2.mcc.CCheckerDetector_create()
    ok = detector.process(img8, cv2.mcc.MCC24, 1)
    if not ok:
        return None, None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None, None
    checker = checkers[0]
    box = checker.getBox()  # 4x2 外枠
    cdraw = cv2.mcc.CCheckerDraw_create(checker)
    # チャートのセル中心はgetColorCharts系APIが直接出さないため、Boxから幾何計算する
    box = np.array(box, dtype=np.float64).reshape(-1, 2)
    return box, checker


def patch_centers_from_box(box: np.ndarray):
    """チャート外枠4点(左上,右上,右下,左下)から 4x6=24 パッチ中心を透視補間で求める。
    OpenCV mcc の Box は [top-left, top-right, bottom-right, bottom-left]。
    ColorChecker Classic は 6列 x 4行。
    """
    tl, tr, br, bl = box[0], box[1], box[2], box[3]
    centers = []
    for r in range(4):        # 行 (上から)
        fy = (r + 0.5) / 4.0
        left = tl + (bl - tl) * fy
        right = tr + (br - tr) * fy
        for c in range(6):    # 列 (左から)
            fx = (c + 0.5) / 6.0
            p = left + (right - left) * fx
            centers.append(p)
    return np.array(centers)  # 24x2


def sample_patch_rgb(img: np.ndarray, center: np.ndarray, box: np.ndarray) -> np.ndarray:
    """パッチ中心の周囲を median サンプリング (パッチサイズの約1/4半径)。"""
    tl, tr, br, bl = box[0], box[1], box[2], box[3]
    # パッチ1個の平均サイズを推定
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2 / 6
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2 / 4
    r = int(max(4, min(w, h) * 0.25))
    cx, cy = int(round(center[0])), int(round(center[1]))
    y0, y1 = max(0, cy - r), min(img.shape[0], cy + r)
    x0, x1 = max(0, cx - r), min(img.shape[1], cx + r)
    roi = img[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(roi, axis=0)


def compute_ccm_rpcc(measured_rgb_linear: np.ndarray, reference_XYZ: np.ndarray, degree: int = 2):
    """Root-Polynomial CCM (Finlayson 2015) を最小二乗で計算。
    measured_rgb_linear: 24x3 (0-1 リニアRGB)
    reference_XYZ:       24x3 (D50 XYZ, 0-1)
    戻り値: 行列 M (3 x n_terms) — apply時は polynomial_expansion 後に M @ expanded
    """
    expanded = np.array([
        polynomial_expansion_Finlayson2015(rgb, degree=degree, root_polynomial_expansion=True)
        for rgb in measured_rgb_linear
    ])  # 24 x n_terms
    # 最小二乗: reference = M @ expanded.T を解く
    M, *_ = np.linalg.lstsq(expanded, reference_XYZ, rcond=None)
    return M.T  # 3 x n_terms


def apply_ccm_rpcc(rgb_linear: np.ndarray, M: np.ndarray, degree: int = 2) -> np.ndarray:
    """RPCC を適用して XYZ を得る。rgb_linear: Nx3"""
    single = rgb_linear.ndim == 1
    rgb = np.atleast_2d(rgb_linear)
    expanded = np.array([
        polynomial_expansion_Finlayson2015(v, degree=degree, root_polynomial_expansion=True)
        for v in rgb
    ])
    XYZ = expanded @ M.T
    return XYZ[0] if single else XYZ


def main():
    cfg = load_config()
    tiff_dir = cfg["linear_tiff_dir"]
    chk_dir = cfg["check_image_dir"]
    ccm_dir = os.path.join(cfg["output_dir"], "ccm")
    os.makedirs(ccm_dir, exist_ok=True)
    os.makedirs(chk_dir, exist_ok=True)

    degree = int(cfg["ccm"].get("degree", 2))
    manual = cfg["chart"].get("manual_corners") or {}

    names, ref_Lab, ref_XYZ, ref_illum = get_reference_lab_d50()

    if len(sys.argv) > 1:
        stems = [os.path.splitext(a)[0] for a in sys.argv[1:]]
        targets = [os.path.join(tiff_dir, s + ".tiff") for s in stems]
    else:
        targets = sorted(glob.glob(os.path.join(tiff_dir, "*.tiff")))

    if not targets:
        print(f"リニアTIFFが見つかりません: {tiff_dir}")
        print("先に 01_decode_raw.py を実行してください")
        sys.exit(1)

    print(f"=== ColorChecker検出 + CCM計算: {len(targets)}ファイル ===")
    summary = []
    for tp in targets:
        base = os.path.splitext(os.path.basename(tp))[0]
        try:
            img16 = tifffile.imread(tp)                     # HxWx3 uint16 リニア
            img_lin = img16.astype(np.float64) / 65535.0    # 0-1 リニア
            # 検出用8bit画像 (ガンマかけて見た目通常化 — mccはガンマ画像の方が検出率が高い)
            img8 = (np.clip(img_lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
            img8_bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

            # --- チャート検出 ---
            box = None
            key_variants = [base, base + ".RW2", base + ".rw2"]
            for k in key_variants:
                if k in manual:
                    box = np.array(manual[k], dtype=np.float64)
                    print(f"  {base}: manual_corners 使用")
                    break
            if box is None:
                box_det, _ = detect_chart_mcc(img8_bgr)
                if box_det is None:
                    print(f"  ❌ {base}: ColorChecker 自動検出失敗 → config.yaml の manual_corners に4隅を記入して再実行してください")
                    summary.append({"file": base, "status": "detect_failed"})
                    continue
                box = box_det

            centers = patch_centers_from_box(box)

            # --- 24パッチ実測RGB (リニア) ---
            measured = np.array([sample_patch_rgb(img_lin, c, box) for c in centers])  # 24x3

            # 白パッチ(19番)で露出正規化: 白の輝度Y ≈ 0.9 (ColorChecker白の反射率)
            # RPCCは露出不変だが、数値安定性のために正規化しておく
            white_rgb = measured[18]
            norm = white_rgb.max()
            if norm <= 0:
                raise ValueError("白パッチのRGBが0です。露出かパッチ位置を確認してください")
            measured_n = measured / norm * 0.9

            # --- CCM計算 (RPCC) ---
            M = compute_ccm_rpcc(measured_n, ref_XYZ, degree=degree)

            # --- 残差評価 (24パッチ) ---
            fitted_XYZ = apply_ccm_rpcc(measured_n, M, degree=degree)
            fitted_Lab = colour.XYZ_to_Lab(fitted_XYZ, illuminant=ref_illum)
            dE = delta_E_CIE2000(ref_Lab, fitted_Lab)
            report = {
                "mean_dE00": float(np.mean(dE)),
                "max_dE00": float(np.max(dE)),
                "worst_patch": names[int(np.argmax(dE))],
            }

            # --- CCM保存 ---
            out = {
                "file": base,
                "degree": degree,
                "white_norm": float(norm),
                "M": M.tolist(),
                "box": box.tolist(),
                "residual": report,
            }
            with open(os.path.join(ccm_dir, base + "_ccm.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1, ensure_ascii=False)

            # --- 確認画像 ---
            vis = img8_bgr.copy()
            cv2.polylines(vis, [box.astype(np.int32)], True, (0, 255, 0), 3)
            for i, c in enumerate(centers):
                cv2.circle(vis, (int(c[0]), int(c[1])), 8, (0, 0, 255), 2)
                cv2.putText(vis, str(i + 1), (int(c[0]) + 10, int(c[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            h, w = vis.shape[:2]
            scale = 2000 / max(h, w)
            if scale < 1:
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(chk_dir, base + "_chart.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            flag = "✅" if report["mean_dE00"] <= 2.0 else ("⚠️" if report["mean_dE00"] <= 4.0 else "❌")
            print(f"  {flag} {base}: CCM ΔE00 平均 {report['mean_dE00']:.2f} / 最大 {report['max_dE00']:.2f} ({report['worst_patch']})")
            summary.append({"file": base, "status": "ok", **report})

        except Exception as e:
            print(f"  ❌ {base}: {e}")
            summary.append({"file": base, "status": "error", "error": str(e)})

    with open(os.path.join(cfg["output_dir"], "ccm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    n_ok = sum(1 for s in summary if s["status"] == "ok")
    print(f"=== 完了: 成功 {n_ok} / {len(summary)} — 確認画像: {chk_dir}/*_chart.jpg ===")


if __name__ == "__main__":
    main()
