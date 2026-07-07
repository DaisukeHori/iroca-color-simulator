#!/usr/bin/env python3
# =========================================================
# 01_decode_raw.py — RW2 → 計測用リニア16bit TIFF 一括変換
# =========================================================
# 使い方:
#   python 01_decode_raw.py                # config.yaml の raw_dir を全件処理
#   python 01_decode_raw.py a05.RW2        # 特定ファイルのみ
#
# 出力:
#   output/linear_tiff/{basename}.tiff     # リニア16bit RGB (計測用)
#   output/check_images/{basename}_preview.jpg  # 目視確認用 (ガンマ2.2適用)
# =========================================================

import sys
import os
import glob
import yaml
import numpy as np

try:
    import rawpy
except ImportError:
    print("ERROR: rawpy がありません。 pip install rawpy を実行してください")
    sys.exit(1)

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile がありません。 pip install tifffile を実行してください")
    sys.exit(1)

import cv2


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decode_one(raw_path: str, cfg: dict) -> str:
    """1つのRW2をリニア16bit TIFFに変換し、確認用JPEGも出力する。
    戻り値: 出力TIFFのパス
    """
    d = cfg["decode"]
    basename = os.path.splitext(os.path.basename(raw_path))[0]

    demosaic_map = {
        "AHD": rawpy.DemosaicAlgorithm.AHD,
        "VNG": rawpy.DemosaicAlgorithm.VNG,
        "PPG": rawpy.DemosaicAlgorithm.PPG,
        "DCB": rawpy.DemosaicAlgorithm.DCB,
    }

    with rawpy.imread(raw_path) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=d["use_camera_wb"],      # カメラWB (太陽光固定) を尊重
            output_bps=d["output_bps"],            # 16bit
            gamma=(d["gamma_power"], d["gamma_slope"]),  # (1,1) = リニア
            no_auto_bright=d["no_auto_bright"],    # 自動明るさ補正OFF
            demosaic_algorithm=demosaic_map.get(d["demosaic"], rawpy.DemosaicAlgorithm.AHD),
            half_size=d["half_size"],
            output_color=rawpy.ColorSpace.sRGB,    # sRGB原色系 (リニア) に変換
            # ハイライト復元はOFF (計測ではクリップをクリップのまま扱う)
            highlight_mode=rawpy.HighlightMode.Clip,
        )

    # 出力: リニアTIFF
    tiff_dir = cfg["linear_tiff_dir"]
    os.makedirs(tiff_dir, exist_ok=True)
    tiff_path = os.path.join(tiff_dir, basename + ".tiff")
    tifffile.imwrite(tiff_path, rgb16, compression="zlib")

    # 出力: 確認用JPEG (ガンマ2.2 で見た目を通常化)
    chk_dir = cfg["check_image_dir"]
    os.makedirs(chk_dir, exist_ok=True)
    preview = (np.clip(rgb16.astype(np.float64) / 65535.0, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    # 長辺2000pxに縮小
    h, w = preview.shape[:2]
    scale = 2000 / max(h, w)
    if scale < 1:
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    preview_path = os.path.join(chk_dir, basename + "_preview.jpg")
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 飽和チェック (計測では致命的なので警告)
    sat_ratio = float((rgb16 >= 65535).any(axis=2).mean())
    flag = ""
    if sat_ratio > 0.005:
        flag = f"  ⚠️ 飽和ピクセル {sat_ratio*100:.1f}% — 露出を下げて再撮影推奨"

    print(f"  ✅ {basename}: {rgb16.shape[1]}x{rgb16.shape[0]} 16bit → {tiff_path}{flag}")
    return tiff_path


def main():
    cfg = load_config()
    raw_dir = cfg["raw_dir"]
    ext = cfg["raw_extension"].lower()

    if len(sys.argv) > 1:
        targets = [os.path.join(raw_dir, a) if not os.path.isabs(a) else a for a in sys.argv[1:]]
    else:
        targets = sorted(
            p for p in glob.glob(os.path.join(raw_dir, "*"))
            if os.path.splitext(p)[1].lower() == ext
        )

    if not targets:
        print(f"RW2ファイルが見つかりません: {raw_dir}/*{ext}")
        print("config.yaml の raw_dir を確認してください")
        sys.exit(1)

    print(f"=== RAW現像開始: {len(targets)}ファイル ===")
    ok, fail = 0, 0
    for t in targets:
        try:
            decode_one(t, cfg)
            ok += 1
        except Exception as e:
            print(f"  ❌ {os.path.basename(t)}: {e}")
            fail += 1
    print(f"=== 完了: 成功 {ok} / 失敗 {fail} ===")


if __name__ == "__main__":
    main()
