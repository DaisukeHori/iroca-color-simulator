#!/usr/bin/env python3
# =========================================================
# test_synthetic.py — 合成データによるパイプライン検証
# =========================================================
# 検証内容:
#   1. 既知のLab色の「毛束」を円筒シェーディング+スペキュラ付きで描画
#   2. ColorChecker 24パッチを既知の公称値で描画
#   3. 疑似カメラ変換 (3x3クロストーク行列 + 露出係数) を全体に適用
#      → 「カメラが撮った歪んだリニアRGB」をシミュレート
#   4. パイプライン (02→03) を通して、既知の真値Labが復元されるか検証
#      復元ΔE00 < 1.0 なら合格
# =========================================================

import os
import json
import numpy as np
import tifffile
import colour
from colour.difference import delta_E_CIE2000
import yaml

np.random.seed(42)

OUT_TIFF_DIR = "./output/linear_tiff"
os.makedirs(OUT_TIFF_DIR, exist_ok=True)

# ---------------------------------------------------------
# 疑似カメラ: 意図的な色クロストーク + 露出
# (実際のGH5センサーが sRGB原色からズレていることを模擬)
# ---------------------------------------------------------
CAMERA_MATRIX = np.array([
    [0.85, 0.12, 0.03],
    [0.08, 0.80, 0.12],
    [0.02, 0.15, 0.83],
])
EXPOSURE = 0.55   # 露出係数 (RPCCの露出不変性テストも兼ねる)


def lab_to_linear_srgb(Lab, illum_xy):
    XYZ = colour.Lab_to_XYZ(np.asarray(Lab, dtype=np.float64), illuminant=illum_xy)
    rgb = colour.XYZ_to_sRGB(XYZ, illuminant=illum_xy, apply_cctf_encoding=False)
    return np.clip(rgb, 0, 1)


def camera_capture(scene_linear_rgb):
    """シーンのリニアsRGB → 疑似カメラRGB"""
    return np.clip(scene_linear_rgb @ CAMERA_MATRIX.T * EXPOSURE, 0, 1)


def build_synthetic_image(sample_lab, illum_xy):
    """1800x2400 の合成画像: 上半分にColorChecker、下半分に円筒毛束"""
    H, W = 1800, 2400
    img = np.full((H, W, 3), 0.35, dtype=np.float64)   # 中間グレー背景

    # --- ColorChecker (公称値で描画) ---
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    names = list(cc.data.keys())
    xyY = np.array([cc.data[n] for n in names])
    XYZ = colour.xyY_to_XYZ(xyY)
    Lab_patches = colour.XYZ_to_Lab(XYZ, illuminant=cc.illuminant)

    chart_x0, chart_y0 = 500, 150
    pw, ph, gap = 220, 160, 14
    corners = None
    for i, lab in enumerate(Lab_patches):
        r, c = divmod(i, 6)
        x0 = chart_x0 + c * (pw + gap)
        y0 = chart_y0 + r * (ph + gap)
        rgb = lab_to_linear_srgb(lab, cc.illuminant)
        img[y0:y0 + ph, x0:x0 + pw] = rgb
    # チャート外枠 (mcc検出は使わずmanual_cornersテストにするため座標を記録)
    corners = [
        [chart_x0, chart_y0],
        [chart_x0 + 6 * pw + 5 * gap, chart_y0],
        [chart_x0 + 6 * pw + 5 * gap, chart_y0 + 4 * ph + 3 * gap],
        [chart_x0, chart_y0 + 4 * ph + 3 * gap],
    ]

    # --- 円筒毛束 (既知Lab、Lambertianシェーディング + スペキュラバンド) ---
    axis_y, R = 1450, 200
    x0, x1 = 300, 2100
    base_rgb = lab_to_linear_srgb(sample_lab, illum_xy)
    for y in range(axis_y - R, axis_y + R + 1):
        s = (y - axis_y) / R          # sin(θ)
        c = np.sqrt(max(0.0, 1 - s * s))   # cos(θ) — Lambert項
        shade = 0.15 + 0.85 * c       # 完全に暗くならないよう floor
        img[y, x0:x1] = base_rgb * shade
    # スペキュラバンド (apexのやや上、太陽反射を模擬)
    spec_y = axis_y - int(R * 0.15)
    for dy in range(-12, 13):
        w_spec = np.exp(-(dy / 6.0) ** 2)
        y = spec_y + dy
        img[y, x0:x1] = np.clip(img[y, x0:x1] + w_spec * 0.55, 0, 1)

    # --- 疑似カメラで撮影 ---
    cam = camera_capture(img)
    # ノイズ (ショットノイズ近似)
    cam = np.clip(cam + np.random.normal(0, 0.002, cam.shape), 0, 1)

    rod = {"axis_y": axis_y, "top_y": axis_y - R, "bottom_y": axis_y + R,
           "x_start": x0 + 50, "x_end": x1 - 50}
    return cam, corners, rod


def main():
    illum_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]

    # 3つのテスト毛束色 (実際のヘアカラー域: 暗褐色 / 赤褐色 / 明るいベージュ)
    tests = {
        "test_dark":  [22.0, 6.0, 9.0],
        "test_red":   [32.0, 22.0, 14.0],
        "test_light": [48.0, 4.0, 20.0],
    }

    manual_corners = {}
    manual_rods = {}
    for name, lab in tests.items():
        cam, corners, rod = build_synthetic_image(lab, illum_xy)
        tiff_path = os.path.join(OUT_TIFF_DIR, name + ".tiff")
        tifffile.imwrite(tiff_path, (cam * 65535).astype(np.uint16), compression="zlib")
        manual_corners[name] = corners
        manual_rods[name] = rod
        print(f"生成: {tiff_path}  (真値Lab = {lab})")

    # config.yaml を合成テスト用に更新
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["chart"]["manual_corners"] = manual_corners
    cfg["cylinder"]["manual_rods"] = manual_rods
    with open("config_test.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print("config_test.yaml を生成 (manual座標入り)")

    # 真値を保存 (検証時に比較)
    with open("./output/synthetic_truth.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=1)


if __name__ == "__main__":
    main()
