# IROCA 視覚色解析パイプライン

i1Pro 2 の計測値（真上スポット測定）と、実際に人間が見る「視覚色」のギャップを埋めるための解析パイプライン。棒巻き毛束を GH5 で撮影し、ColorChecker 補正を通して角度別の視覚 Lab を取得、i1Pro 2 → 視覚色の補正関数を学習する。

## 検証状況

合成テストデータ（既知Lab の円筒毛束 + 疑似カメラ歪み + スペキュラバンド）でエンドツーエンド検証済み:

| 検証項目 | 結果 |
|---|---|
| RPCC による疑似カメラ補正 (クロストーク + 露出0.55) | 24パッチ残差 ΔE00 平均 0.31 ✅ |
| 円筒毛束の真値Lab復元 (apex 0°) | ΔE00 0.23〜0.54 ✅ |
| スペキュラバンド除去 | 機能確認 ✅ |
| 補正関数学習 + LOOCV (合成40ペア, ノイズσ=0.8) | LOOCV平均 ΔE00 1.09 ✅ |

## セットアップ

```bash
cd analysis
pip install -r requirements.txt
```

Python 3.10 以上。Windows / Mac / Linux 対応。

## 撮影プロトコル

### 機材
- Panasonic GH5 ×1台 + 三脚
- 黒マット棒（直径3cm × 長さ15cm）— 毛束を巻き付ける
- X-Rite ColorChecker Classic
- マスキングテープ（毛束固定用）

### カメラ設定
```
モード: マニュアル / ISO: 100 / 絞り: f8
シャッター: RAWヒストグラムで白パッチが飽和しない値 (太陽光下 1/100〜1/400)
WB: 太陽光 5500K に固定（必須。オートWB禁止）
フォーカス: マニュアル、毛束中央
記録: RAW (RW2) のみ
手ぶれ補正: OFF（三脚使用）
```

### 配置
```
- 毛束を棒に巻く（毛流れは棒の軸方向 = 縦、両端をテープ固定）
- 棒を水平に置く（画像内で棒が横方向に写るように）
- ColorChecker を棒の隣に同一フレーム内・同じ照明下で配置
- カメラ距離 約1m、棒の真横から撮影
- 光源: 快晴の正午（太陽が真上）が理想。曇天・高演色LED(CRI90+)でも可
- 影が毛束・チャートにかからないこと
```

### ファイル命名
```
{sample_id}.RW2     例: a05.RW2, a27.RW2
```

40サンプルで撮影時間の目安 40〜50分（巻き替え含む）。

## 実行手順

```bash
# 0. RW2ファイルを analysis/raw/ に置く

# 1. RAW → リニア16bit TIFF
python 01_decode_raw.py
#    → output/linear_tiff/*.tiff, output/check_images/*_preview.jpg
#    ⚠️ 飽和警告が出たファイルは露出を下げて再撮影

# 2. ColorChecker検出 + CCM計算
python 02_detect_chart.py
#    → output/ccm/*_ccm.json, output/check_images/*_chart.jpg
#    24パッチの残差ΔE00が表示される (平均2以下が正常)
#    自動検出に失敗したら config.yaml の chart.manual_corners に4隅座標を記入して再実行

# 3. 棒巻き毛束の角度別Lab抽出
python 03_extract_cylinder.py
#    → output/visual_labs.csv, output/check_images/*_cylinder.jpg
#    確認画像で角度帯(赤線)が毛束に正しく載っているか目視確認
#    ズレていたら config.yaml の cylinder.manual_rods に座標記入して再実行

# 4. 補正関数の学習 + LOOCV精度検証
python 04_learn_correction.py
#    → output/correction_model.json
#    LOOCV平均ΔE00: ≤3 良好 / 3〜5 実用可 / >5 要改善(サンプル追加検討)

# 5. Supabaseへ保存 (要テーブル作成、下記SQL参照)
python 05_upload_supabase.py            # dry-run
python 05_upload_supabase.py --commit   # 実書き込み
```

## Supabase 事前準備 (SQL Editor で実行)

```sql
CREATE TABLE IF NOT EXISTS iroca_calibration_shots (
  id BIGSERIAL PRIMARY KEY,
  sample_name TEXT NOT NULL,
  angles_deg JSONB,
  visual_labs JSONB,
  ccm_residual JSONB,
  shot_metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_calib_shots_sample ON iroca_calibration_shots(sample_name);
```

## シミュレータ統合

`output/correction_model.json` を `public/` にコピーし、`iroca-calibration.js` から読み込む。
モデル形式:

```json
{
  "type": "affine_per_angle",
  "angles_deg": [0, 15, 30, 45, 60],
  "per_angle": {
    "0":  {"A": [[...3x3...]], "b": [x,y,z]},
    "45": {"A": [[...3x3...]], "b": [x,y,z]}
  }
}
```

適用: `visual_Lab = A @ i1pro_Lab + b` （角度ごと）

## 技術メモ

- **RPCC (Root-Polynomial, Finlayson 2015)** を採用。通常の多項式補正と違い露出不変
  なので、屋外撮影の露出バラつきに強い
- **スペキュラ除去**: 各角度帯ROI内の輝度上位20%をカットしてから median
- **円筒幾何**: 画像Y座標と法線角の関係 y = axis_y + R·sin(θ)。±両側を平均
- 合成検証は `test_synthetic.py` で再現可能（`config_test.yaml` が生成される）
