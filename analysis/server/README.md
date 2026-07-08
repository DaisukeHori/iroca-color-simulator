# IROCA Vision API — 運用ガイド

写真をアップロードするだけで「実データ処理 → 補正モデル学習 → シミュレーター反映」まで自動で回るシステム。

## 全体構成

```
[GH5で撮影: 棒巻き毛束 + ColorChecker]
        ↓ RW2/JPG をブラウザでアップ
[calibrate.html]  (GitHub Pages: /calibrate.html)
        ↓ HTTPS
[IROCA Vision API]  (LXC + FastAPI, Cloudflare Tunnel公開)
        ↓ 自動: リニア現像 → チャート検出 → CCM → 円筒角度解析
[Supabase]  iroca_calibration_shots (視覚Lab) / iroca_correction_model (学習モデル)
        ↓ シミュレーターが起動時に自動fetch (1hキャッシュ)
[index.html]  「👁 見え方」ボタンで 計測値/0°/15°/30°/45°/60° を切替表示
```

## 初回セットアップ

### 1. Supabase テーブル作成 (SQL Editor で1回だけ)

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

CREATE TABLE IF NOT EXISTS iroca_correction_model (
  id BIGSERIAL PRIMARY KEY,
  model JSONB NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. LXC にサーバーをデプロイ

```bash
# LXC (Ubuntu 24.04, 2GB RAM以上推奨) 内で:
mkdir -p /opt/iroca-vision && cd /opt/iroca-vision
# server.py, requirements.txt, setup.sh を配置して:
sudo bash setup.sh
# → systemd サービス iroca-vision が port 8340 で起動
```

APIキーを変える場合: `/etc/systemd/system/iroca-vision.service` の
`Environment=API_KEY=...` を編集して `systemctl restart iroca-vision`

### 3. Cloudflare Tunnel で公開

既存の cloudflared に ingress を1行追加 (例: iroca-vision.appserver.tokyo → http://<LXCのIP>:8340)。

### 4. calibrate.html の接続設定

https://daisukehori.github.io/iroca-color-simulator/calibrate.html を開き、
サーバーURLとAPIキーを入力して「保存して接続テスト」。設定はブラウザに保存される。

## 日常の使い方 (撮影当日)

1. 撮影: 棒巻き毛束 + ColorChecker を同一フレームで撮る (RAW推奨、JPGも可)
2. calibrate.html でサンプルID (例: a05) を入れて写真をドロップ
3. 自動検出結果が画像に重なって表示される
   - 緑の4点 = チャート四隅 → ズレていればドラッグ修正
   - 青の枠 = 毛束範囲、緑の横線 = 棒の中心 → 同様にドラッグ修正
4. 「色を抽出して保存」→ 角度別スウォッチとΔE00が表示され、DBに保存
5. 全サンプル終わったら「補正モデルを学習」→ LOOCV精度表が出て、
   シミュレーターに自動反映 (index.html の「👁 見え方」ボタンが有効化)

## トラブルシューティング (自己解決ガイド)

すべてのエラーは画面に「何が起きたか + どうすれば直るか」が表示されます。代表例:

| 症状 | 原因 | 対処 |
|---|---|---|
| チャート自動検出失敗 | ブレ/小さすぎ/極端な角度 | 画面上で4隅を手動ドラッグ (そのまま進める) |
| チャートΔE00 > 4 | 照明の演色性不足/影/色かぶり | 直射日光 or CRI90+ LED で再撮影 |
| 飽和警告 | 露出オーバー | シャッターを1〜2段速くして再撮影 |
| 角度帯ピクセル不足 | 毛束が細い/範囲が狭い | 青枠を広げる or 近接撮影し直し |
| LOOCV > 5 | サンプル不足/外れ撮影 | 表示された最悪サンプルを撮り直し、またはPhase B代表を追加 |
| DB保存失敗 | ネットワーク | 「色を抽出して保存」をもう一度押すだけでOK (再アップ不要) |
| ジョブ期限切れ (404) | アップから24時間経過 | 再アップロード |
| サーバー無応答 | LXC/Tunnel停止 | `systemctl status iroca-vision` / cloudflared を確認 |

## 精度の判定基準

- チャート補正 ΔE00 平均: **2以下=良好** / 2〜4=可 / 4超=撮影条件を改善
- 学習LOOCV ΔE00 平均: **3以下=良好** / 3〜5=実用可 / 5超=要改善

## 技術仕様

- カメラ補正: Root-Polynomial CCM (Finlayson 2015, 露出不変, degree=2)
- 円筒幾何: y = axis_y + R·sin(θ)、角度帯±6°、±両側平均
- スペキュラ除去: 帯内輝度上位20%カット → median
- 補正モデル: 角度別アフィン (visual_Lab = A·i1pro_Lab + b)、LOOCV検証付き
- 合成データ検証済み: カメラ歪み+露出0.55+スペキュラ入りで真値復元 ΔE00 0.23〜0.54
- JS適用側とPython学習側の行列演算一致を検証済み
