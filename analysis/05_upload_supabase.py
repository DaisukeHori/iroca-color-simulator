#!/usr/bin/env python3
# =========================================================
# 05_upload_supabase.py — 視覚Lab結果を Supabase に保存
# =========================================================
# 使い方:
#   SUPABASE_KEY=<key> python 05_upload_supabase.py           # dry-run (確認のみ)
#   SUPABASE_KEY=<key> python 05_upload_supabase.py --commit  # 実書き込み
#
# 事前に Supabase SQL Editor で以下を実行しておくこと:
#
#   CREATE TABLE IF NOT EXISTS iroca_calibration_shots (
#     id BIGSERIAL PRIMARY KEY,
#     sample_name TEXT NOT NULL,
#     angles_deg JSONB,
#     visual_labs JSONB,        -- {"0":[L,a,b], "15":[L,a,b], ...}
#     ccm_residual JSONB,       -- {"mean_dE00":..,"max_dE00":..}
#     shot_metadata JSONB,
#     created_at TIMESTAMPTZ DEFAULT NOW()
#   );
# =========================================================

import os
import sys
import csv
import json
import requests
import yaml

SUPA_URL = "https://flmeolcfutuwwbjmzyoz.supabase.co"
SUPA_KEY_DEFAULT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZsbWVvbGNmdXR1d3diam16eW96Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM5NzAxODYsImV4cCI6MjA3OTU0NjE4Nn0."
                    "VVxUxKexNeN6dUiAMDkCNlnIoXa-F5rfBqHPBDcwdnU")


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    commit = "--commit" in sys.argv
    cfg = load_config()
    angles = cfg["cylinder"]["angles_deg"]
    key = os.environ.get("SUPABASE_KEY", SUPA_KEY_DEFAULT)

    csv_path = os.path.join(cfg["output_dir"], "visual_labs.csv")
    ccm_summary_path = os.path.join(cfg["output_dir"], "ccm_summary.json")

    if not os.path.exists(csv_path):
        print(f"visual_labs.csv がありません: {csv_path}")
        sys.exit(1)

    ccm_map = {}
    if os.path.exists(ccm_summary_path):
        with open(ccm_summary_path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                if row.get("status") == "ok":
                    ccm_map[row["file"]] = {"mean_dE00": row["mean_dE00"], "max_dE00": row["max_dE00"]}

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row["sample"]
            labs = {}
            for a in angles:
                try:
                    labs[str(a)] = [float(row[f"L_{a}"]), float(row[f"a_{a}"]), float(row[f"b_{a}"])]
                except (ValueError, KeyError):
                    pass
            records.append({
                "sample_name": s,
                "angles_deg": angles,
                "visual_labs": labs,
                "ccm_residual": ccm_map.get(s),
                "shot_metadata": {"pipeline_version": "1.0"},
            })

    print(f"アップロード対象: {len(records)}件")
    for r in records[:5]:
        print(f"  {r['sample_name']}: 角度{list(r['visual_labs'].keys())}")
    if len(records) > 5:
        print(f"  ... 他{len(records)-5}件")

    if not commit:
        print("\n★ dry-run です。実書き込みは --commit を付けて実行してください")
        return

    headers = {"apikey": key, "Authorization": f"Bearer {key}",
               "Content-Type": "application/json", "Prefer": "return=minimal"}
    ok, fail = 0, 0
    for r in records:
        resp = requests.post(f"{SUPA_URL}/rest/v1/iroca_calibration_shots",
                             headers=headers, json=r, timeout=30)
        if resp.status_code in (200, 201):
            ok += 1
        else:
            fail += 1
            print(f"  ❌ {r['sample_name']}: HTTP {resp.status_code} {resp.text[:120]}")
    print(f"完了: 成功 {ok} / 失敗 {fail}")


if __name__ == "__main__":
    main()
