#!/usr/bin/env python3
# =========================================================
# 04_learn_correction.py — i1Pro2→視覚Lab 補正関数の学習 + LOOCV検証
# =========================================================
# 使い方:
#   SUPABASE_KEY=<anon key> python 04_learn_correction.py
#   (キー未設定時はスクリプト内デフォルトの anon key を使用)
#
# 処理内容:
#   1. Supabase iroca_sample_summary から i1Pro2 の median Lab を取得
#   2. output/visual_labs.csv (03の出力) と sample名で結合
#   3. 各角度について アフィン変換 (3x3行列 + バイアス) を最小二乗で学習
#        visual_Lab ≈ A @ i1pro_Lab + b     (パラメータ12個/角度)
#   4. Leave-One-Out 交差検証で 未知サンプルへの予測ΔE00 を評価
#   5. output/correction_model.json にモデル出力 (シミュレータが読む)
# =========================================================

import os
import sys
import csv
import json
import numpy as np
import requests
import colour
from colour.difference import delta_E_CIE2000
import yaml

SUPA_URL = "https://flmeolcfutuwwbjmzyoz.supabase.co"
SUPA_KEY_DEFAULT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZsbWVvbGNmdXR1d3diam16eW96Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM5NzAxODYsImV4cCI6MjA3OTU0NjE4Nn0."
                    "VVxUxKexNeN6dUiAMDkCNlnIoXa-F5rfBqHPBDcwdnU")


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_i1pro_labs(key: str) -> dict:
    """iroca_sample_summary から {sample_name: [L,a,b]} を取得 (ページング対応)。"""
    out = {}
    frm = 0
    while True:
        r = requests.get(
            f"{SUPA_URL}/rest/v1/iroca_sample_summary",
            params={"select": "sample_name,median_lab_l,median_lab_a,median_lab_b"},
            headers={"apikey": key, "Authorization": f"Bearer {key}",
                     "Range": f"{frm}-{frm+999}"},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        for row in batch:
            if row["median_lab_l"] is not None:
                out[row["sample_name"]] = [row["median_lab_l"], row["median_lab_a"], row["median_lab_b"]]
        if len(batch) < 1000:
            break
        frm += 1000
    return out


def load_visual_csv(path: str, angles):
    """visual_labs.csv → {sample: {angle: [L,a,b]}}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row["sample"]
            per = {}
            for a in angles:
                try:
                    per[a] = [float(row[f"L_{a}"]), float(row[f"a_{a}"]), float(row[f"b_{a}"])]
                except (ValueError, KeyError):
                    per[a] = None
            out[s] = per
    return out


def fit_affine(X: np.ndarray, Y: np.ndarray):
    """Y ≈ A X + b を最小二乗で解く。X,Y: Nx3。戻り値 (A 3x3, b 3)"""
    N = X.shape[0]
    Xh = np.hstack([X, np.ones((N, 1))])          # Nx4
    W, *_ = np.linalg.lstsq(Xh, Y, rcond=None)    # 4x3
    A = W[:3].T                                   # 3x3
    b = W[3]                                      # 3
    return A, b


def loocv_dE(X: np.ndarray, Y: np.ndarray):
    """Leave-One-Out 交差検証。各サンプルを外して学習→予測→ΔE00。戻り値: ΔE配列"""
    N = X.shape[0]
    dEs = np.zeros(N)
    for i in range(N):
        mask = np.arange(N) != i
        A, b = fit_affine(X[mask], Y[mask])
        pred = A @ X[i] + b
        dEs[i] = delta_E_CIE2000(Y[i], pred)
    return dEs


def main():
    cfg = load_config()
    angles = cfg["cylinder"]["angles_deg"]
    key = os.environ.get("SUPABASE_KEY", SUPA_KEY_DEFAULT)

    csv_path = os.path.join(cfg["output_dir"], "visual_labs.csv")
    if not os.path.exists(csv_path):
        print(f"visual_labs.csv がありません: {csv_path}")
        print("先に 03_extract_cylinder.py を実行してください")
        sys.exit(1)

    print("=== i1Pro2 データ取得中 (Supabase) ===")
    i1 = fetch_i1pro_labs(key)
    print(f"  i1Pro2 サマリー: {len(i1)}件")

    visual = load_visual_csv(csv_path, angles)
    print(f"  視覚Lab: {len(visual)}件")

    # ペア構築
    pairs = {a: {"X": [], "Y": [], "names": []} for a in angles}
    unmatched = []
    for s, per in visual.items():
        if s not in i1:
            unmatched.append(s)
            continue
        for a in angles:
            if per.get(a) is not None:
                pairs[a]["X"].append(i1[s])
                pairs[a]["Y"].append(per[a])
                pairs[a]["names"].append(s)
    if unmatched:
        print(f"  ⚠️ i1Pro2側に見つからないサンプル: {unmatched}")

    model = {"type": "affine_per_angle", "illuminant": cfg["colorimetry"]["illuminant"],
             "angles_deg": angles, "per_angle": {}, "validation": {}}

    print("\n=== 角度別 補正関数の学習 + LOOCV ===")
    print(f"{'角度':>5} | {'n':>3} | {'学習残差ΔE00':>12} | {'LOOCV平均':>9} | {'LOOCV最大':>9} | 判定")
    print("-" * 65)
    for a in angles:
        X = np.array(pairs[a]["X"], dtype=np.float64)
        Y = np.array(pairs[a]["Y"], dtype=np.float64)
        n = len(X)
        if n < 8:
            print(f"{a:>4}° | {n:>3} | サンプル不足 (最低8件必要)")
            continue
        A, b = fit_affine(X, Y)
        fit_pred = X @ A.T + b
        fit_dE = delta_E_CIE2000(Y, fit_pred)
        cv_dE = loocv_dE(X, Y)
        mean_cv, max_cv = float(np.mean(cv_dE)), float(np.max(cv_dE))
        worst = pairs[a]["names"][int(np.argmax(cv_dE))]
        verdict = "✅ 良好" if mean_cv <= 3 else ("⚠️ 実用可" if mean_cv <= 5 else "❌ 要改善")
        print(f"{a:>4}° | {n:>3} | {np.mean(fit_dE):>12.2f} | {mean_cv:>9.2f} | {max_cv:>9.2f} | {verdict} (最悪:{worst})")

        model["per_angle"][str(a)] = {"A": A.tolist(), "b": b.tolist(), "n_samples": n}
        model["validation"][str(a)] = {
            "fit_mean_dE00": float(np.mean(fit_dE)),
            "loocv_mean_dE00": mean_cv,
            "loocv_max_dE00": max_cv,
            "loocv_worst_sample": worst,
            "loocv_dE00_per_sample": {nm: round(float(d), 2)
                                      for nm, d in zip(pairs[a]["names"], cv_dE)},
        }

    out_path = os.path.join(cfg["output_dir"], "correction_model.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1, ensure_ascii=False)
    print(f"\n=== モデル出力: {out_path} ===")
    print("シミュレータ統合: このJSONを public/ にコピーして iroca-calibration.js から読み込みます")


if __name__ == "__main__":
    main()
