#!/usr/bin/env python3
# =========================================================
# server.py — IROCA 視覚色解析 API サーバー
# =========================================================
# 検証済みパイプライン (01〜05) をWeb API化したもの。
# calibrate.html からアップロード → 自動処理 → 失敗時は
# ブラウザ上の手動指定で続行できる2段階API。
#
# 起動:
#   API_KEY=<好きな文字列> uvicorn server:app --host 0.0.0.0 --port 8340
#
# エンドポイント:
#   POST /api/upload    画像アップロード + 自動検出 (チャート/棒)
#   POST /api/extract   座標確定 → 角度別Lab抽出 → Supabase保存
#   POST /api/learn     全データから補正モデル学習 → Supabase保存
#   GET  /api/samples   処理済みサンプル一覧
#   GET  /api/image/{job_id}/{kind}   プレビュー/確認画像
#   GET  /api/health    ヘルスチェック
# =========================================================

import io
import os
import json
import time
import uuid
import shutil
import traceback
import numpy as np
import cv2
import requests as rq

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

import colour
from colour.characterisation import polynomial_expansion_Finlayson2015
from colour.difference import delta_E_CIE2000

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "iroca-vision-2026")
JOB_DIR = os.environ.get("JOB_DIR", "/opt/iroca-vision/jobs")
JOB_TTL_SEC = 24 * 3600          # ジョブ保持 24時間
SUPA_URL = "https://flmeolcfutuwwbjmzyoz.supabase.co"
SUPA_KEY = os.environ.get("SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZsbWVvbGNmdXR1d3diam16eW96Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM5NzAxODYsImV4cCI6MjA3OTU0NjE4Nn0."
    "VVxUxKexNeN6dUiAMDkCNlnIoXa-F5rfBqHPBDcwdnU")

ANGLES = [0, 15, 30, 45, 60]
BAND_HALFWIDTH_DEG = 6
SPECULAR_CUT_PCT = 80
MIN_PIXELS = 200
RPCC_DEGREE = 2
ILLUM_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]

os.makedirs(JOB_DIR, exist_ok=True)

app = FastAPI(title="IROCA Vision API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # GitHub Pages から叩くため
    allow_methods=["*"],
    allow_headers=["*"],
)


def auth(x_api_key):
    if x_api_key != API_KEY:
        raise HTTPException(401, "APIキーが違います (X-API-Key ヘッダ)")


# ---------------------------------------------------------
# 診断メッセージ辞書 — 「何が起きたか + どうすれば直るか」
# ---------------------------------------------------------
def diag(code, **kw):
    """診断メッセージ生成。lambda遅延評価 (該当コードのみフォーマット)。"""
    D = {
        "saturation": lambda: {
            "level": "warning",
            "message": f"飽和ピクセルが {kw.get('pct',0):.1f}% あります。白パッチや毛束のハイライトが白飛びしている可能性があります。",
            "fix": "露出を1〜2段下げて再撮影してください (シャッター速度を速く)。このまま続行も可能ですが、明るい色の精度が落ちます。",
        },
        "underexposed": lambda: {
            "level": "warning",
            "message": f"画像全体が暗すぎます (平均輝度 {kw.get('mean',0):.3f})。ノイズが増えて色精度が落ちます。",
            "fix": "露出を上げて再撮影を推奨します (シャッターを遅く / ISOはなるべく100のまま)。",
        },
        "chart_not_found": lambda: {
            "level": "action_required",
            "message": "ColorCheckerを自動検出できませんでした。",
            "fix": "画面上でチャートの4隅 (左上→右上→右下→左下) をクリックして指定してください。チャートが小さすぎる/ブレている場合は再撮影してください。",
        },
        "chart_low_quality": lambda: {
            "level": "warning",
            "message": f"カラーチャート補正の残差が大きめです (ΔE00平均 {kw.get('de',0):.2f})。照明の演色性が低いか、チャートに影や色かぶりがある可能性があります。",
            "fix": "直射日光下または高演色LED (CRI90+) で、チャートに影がかからないように再撮影すると改善します。ΔE00平均2以下が理想です。",
        },
        "rod_not_found": lambda: {
            "level": "action_required",
            "message": "棒巻き毛束の位置を自動検出できませんでした。",
            "fix": "画面上で毛束の上端・中心・下端の3本の線と左右範囲をドラッグして指定してください。",
        },
        "band_too_small": lambda: {
            "level": "warning",
            "message": f"{kw.get('angle')}°の角度帯のピクセル数が不足しています ({kw.get('n')}px)。",
            "fix": "毛束が細すぎるか、解析範囲(左右)が狭すぎます。カメラを近づけて毛束を大きく撮るか、x範囲を広げてください。",
        },
        "duplicate_sample": lambda: {
            "level": "info",
            "message": f"サンプル {kw.get('sample')} は既に処理済みです。今回のデータで上書き保存します。",
            "fix": "",
        },
        "supabase_error": lambda: {
            "level": "error",
            "message": f"Supabaseへの保存に失敗しました: {kw.get('detail','')}",
            "fix": "ネットワークを確認して再実行してください。データはサーバー上に残っているので再アップロードは不要です (「保存リトライ」を押す)。",
        },
        "insufficient_samples": lambda: {
            "level": "error",
            "message": f"学習には最低8サンプル必要です (現在 {kw.get('n')}件)。",
            "fix": "追加のサンプルをアップロードしてから再度学習してください。",
        },
        "loocv_poor": lambda: {
            "level": "warning",
            "message": f"交差検証の予測精度が低めです ({kw.get('angle')}°: 平均ΔE00 {kw.get('de',0):.2f})。",
            "fix": f"最悪サンプル ({kw.get('worst','?')}) の撮影を見直すか、Phase B の代表サンプルを5-10件追加撮影すると改善する可能性があります。",
        },
        "i1pro_missing": lambda: {
            "level": "warning",
            "message": f"i1Pro2側の計測データが見つからないサンプル: {kw.get('samples')}",
            "fix": "サンプルIDのスペルを確認してください。measure.html で計測済みのplan ID (例: a05) と一致させる必要があります。",
        },
    }
    d = D[code]()
    d["code"] = code
    return d


# ---------------------------------------------------------
# 画像デコード (RW2 / JPEG / TIFF / PNG 対応)
# ---------------------------------------------------------
def decode_upload(path: str):
    """アップロードファイル → リニアRGB float (0-1)。
    RW2: rawpy でリニア現像。
    JPEG/PNG: sRGBガンマを外してリニア化 (8bit なので精度は落ちるが動く)。
    TIFF: 16bit ならリニアとみなす (01_decode_raw.py の出力互換)。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".rw2", ".raw", ".dng", ".nef", ".cr2", ".arw"):
        import rawpy
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True, output_bps=16, gamma=(1, 1),
                no_auto_bright=True, output_color=rawpy.ColorSpace.sRGB,
                highlight_mode=rawpy.HighlightMode.Clip)
        return rgb16.astype(np.float64) / 65535.0, "raw_linear"
    if ext in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(path)
        if img.dtype == np.uint16:
            return img.astype(np.float64) / 65535.0, "tiff16_linear"
        return colour.cctf_decoding(img.astype(np.float64) / 255.0), "tiff8_srgb"
    # JPEG / PNG: sRGB → リニア
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"画像として読めません: {ext} — RW2/JPG/PNG/TIFFに対応しています")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return colour.cctf_decoding(rgb), "jpeg_srgb"


# ---------------------------------------------------------
# ColorChecker (02_detect_chart.py 検証済みロジック)
# ---------------------------------------------------------
def get_reference():
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    names = list(cc.data.keys())
    xyY = np.array([cc.data[n] for n in names])
    XYZ = colour.xyY_to_XYZ(xyY)
    Lab = colour.XYZ_to_Lab(XYZ, illuminant=cc.illuminant)
    return names, Lab, XYZ, cc.illuminant

REF_NAMES, REF_LAB, REF_XYZ, REF_ILLUM = get_reference()


def detect_chart_auto(img_lin):
    img8 = (np.clip(img_lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    try:
        det = cv2.mcc.CCheckerDetector_create()
        if det.process(bgr, cv2.mcc.MCC24, 1):
            lst = det.getListColorChecker()
            if lst:
                box = np.array(lst[0].getBox(), dtype=np.float64).reshape(-1, 2)
                return box
    except Exception:
        pass
    return None


def patch_centers(box):
    tl, tr, br, bl = box[0], box[1], box[2], box[3]
    out = []
    for r in range(4):
        fy = (r + 0.5) / 4.0
        left, right = tl + (bl - tl) * fy, tr + (br - tr) * fy
        for c in range(6):
            fx = (c + 0.5) / 6.0
            out.append(left + (right - left) * fx)
    return np.array(out)


def sample_patch(img, center, box):
    tl, tr, br, bl = box[0], box[1], box[2], box[3]
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2 / 6
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2 / 4
    r = int(max(4, min(w, h) * 0.25))
    cx, cy = int(round(center[0])), int(round(center[1]))
    roi = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r].reshape(-1, 3)
    return np.median(roi, axis=0)


def compute_ccm(img_lin, box):
    centers = patch_centers(box)
    measured = np.array([sample_patch(img_lin, c, box) for c in centers])
    white = measured[18]
    norm = float(white.max())
    if norm <= 1e-6:
        raise ValueError("白パッチのRGBがゼロです。チャート位置か露出を確認してください")
    mn = measured / norm * 0.9
    expanded = np.array([polynomial_expansion_Finlayson2015(v, degree=RPCC_DEGREE,
                         root_polynomial_expansion=True) for v in mn])
    M, *_ = np.linalg.lstsq(expanded, REF_XYZ, rcond=None)
    M = M.T
    fitted = expanded @ M.T
    fLab = colour.XYZ_to_Lab(fitted, illuminant=REF_ILLUM)
    dE = delta_E_CIE2000(REF_LAB, fLab)
    return M, norm, {"mean_dE00": float(np.mean(dE)), "max_dE00": float(np.max(dE)),
                     "worst_patch": REF_NAMES[int(np.argmax(dE))]}


def apply_ccm(rgb, M):
    single = np.asarray(rgb).ndim == 1
    arr = np.atleast_2d(rgb)
    exp = np.array([polynomial_expansion_Finlayson2015(v, degree=RPCC_DEGREE,
                    root_polynomial_expansion=True) for v in arr])
    XYZ = exp @ np.asarray(M).T
    return XYZ[0] if single else XYZ


# ---------------------------------------------------------
# 円筒解析 (03_extract_cylinder.py 検証済みロジック)
# ---------------------------------------------------------
def detect_rod_auto(img_lin):
    h, w = img_lin.shape[:2]
    lum = img_lin @ np.array([0.2126, 0.7152, 0.0722])
    x0, x1 = int(w * 0.2), int(w * 0.8)
    row = lum[:, x0:x1].mean(axis=1)
    thr = np.median(row) * 0.6
    dark = row < thr
    best_len, best_start, cur, cs = 0, -1, 0, -1
    for i, v in enumerate(dark):
        if v:
            if cur == 0:
                cs = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, cs
        else:
            cur = 0
    if best_len < h * 0.03:
        return None
    return {"axis_y": int(best_start + best_len // 2), "top_y": int(best_start),
            "bottom_y": int(best_start + best_len - 1), "x_start": x0, "x_end": x1}


def extract_cylinder(img_lin, rod, M, white_norm):
    axis_y = rod["axis_y"]
    R = (rod["bottom_y"] - rod["top_y"]) / 2.0
    x0, x1 = rod["x_start"], rod["x_end"]
    half = np.deg2rad(BAND_HALFWIDTH_DEG)
    lw = np.array([0.2126, 0.7152, 0.0722])
    labs, warns = {}, []
    for a in ANGLES:
        th = np.deg2rad(a)
        chunks = []
        for s in ([th] if a == 0 else [th, -th]):
            ylo = axis_y + R * np.sin(s - half)
            yhi = axis_y + R * np.sin(s + half)
            ylo, yhi = int(round(min(ylo, yhi))), int(round(max(ylo, yhi)))
            ylo, yhi = max(rod["top_y"], ylo), min(rod["bottom_y"], yhi)
            if yhi > ylo:
                chunks.append(img_lin[ylo:yhi + 1, x0:x1].reshape(-1, 3))
        if not chunks:
            labs[str(a)] = None
            continue
        px = np.concatenate(chunks)
        keep = px[(px @ lw) <= np.percentile(px @ lw, SPECULAR_CUT_PCT)]
        if len(keep) < MIN_PIXELS:
            labs[str(a)] = None
            warns.append(diag("band_too_small", angle=a, n=len(keep)))
            continue
        rgb = np.median(keep, axis=0) / white_norm * 0.9
        XYZ = apply_ccm(rgb, M)
        Lab = colour.XYZ_to_Lab(XYZ, illuminant=ILLUM_XY)
        labs[str(a)] = [round(float(v), 2) for v in Lab]
    return labs, warns


# ---------------------------------------------------------
# 可視化
# ---------------------------------------------------------
def render_check(img_lin, box, rod, max_px=1800):
    img8 = (np.clip(img_lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    vis = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    if box is not None:
        cv2.polylines(vis, [np.asarray(box, np.int32)], True, (0, 255, 0), 3)
        for i, c in enumerate(patch_centers(np.asarray(box))):
            cv2.circle(vis, (int(c[0]), int(c[1])), 7, (0, 0, 255), 2)
    if rod is not None:
        axis_y, R = rod["axis_y"], (rod["bottom_y"] - rod["top_y"]) / 2.0
        x0, x1 = rod["x_start"], rod["x_end"]
        cv2.rectangle(vis, (x0, rod["top_y"]), (x1, rod["bottom_y"]), (255, 200, 0), 2)
        cv2.line(vis, (x0, axis_y), (x1, axis_y), (0, 255, 0), 2)
        for a in ANGLES:
            for s in ([1] if a == 0 else [1, -1]):
                y = int(round(axis_y + R * np.sin(np.deg2rad(a * s))))
                cv2.line(vis, (x0, y), (x1, y), (0, 0, 255), 1)
    h, w = vis.shape[:2]
    sc = max_px / max(h, w)
    if sc < 1:
        vis = cv2.resize(vis, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), (sc if sc < 1 else 1.0)


# ---------------------------------------------------------
# ジョブ管理
# ---------------------------------------------------------
def job_path(job_id, name=""):
    d = os.path.join(JOB_DIR, job_id)
    return os.path.join(d, name) if name else d


def cleanup_jobs():
    now = time.time()
    for j in os.listdir(JOB_DIR):
        p = os.path.join(JOB_DIR, j)
        try:
            if os.path.isdir(p) and now - os.path.getmtime(p) > JOB_TTL_SEC:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------
# API
# ---------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0", "angles": ANGLES}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), sample_id: str = Form(...),
                 x_api_key: str = Header(None)):
    auth(x_api_key)
    cleanup_jobs()
    sample_id = sample_id.strip().lower()
    if not sample_id:
        raise HTTPException(400, "sample_id が空です")
    job_id = uuid.uuid4().hex[:12]
    os.makedirs(job_path(job_id), exist_ok=True)
    raw_path = job_path(job_id, "input" + os.path.splitext(file.filename or "x.jpg")[1].lower())
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    warnings = []
    try:
        img_lin, source_kind = decode_upload(raw_path)
    except Exception as e:
        shutil.rmtree(job_path(job_id), ignore_errors=True)
        raise HTTPException(422, f"デコード失敗: {e}")

    # 画質診断
    sat = float((img_lin >= 0.999).all(axis=2).mean())
    if sat > 0.005:
        warnings.append(diag("saturation", pct=sat * 100))
    mean_lum = float((img_lin @ np.array([0.2126, 0.7152, 0.0722])).mean())
    if mean_lum < 0.02:
        warnings.append(diag("underexposed", mean=mean_lum))

    # 自動検出
    box = detect_chart_auto(img_lin)
    if box is None:
        warnings.append(diag("chart_not_found"))
    rod = detect_rod_auto(img_lin)
    if rod is None:
        warnings.append(diag("rod_not_found"))

    # リニア画像をジョブに保存 (extract で再利用)
    np.save(job_path(job_id, "linear.npy"), img_lin.astype(np.float32))
    meta = {"sample_id": sample_id, "source_kind": source_kind,
            "img_h": img_lin.shape[0], "img_w": img_lin.shape[1]}
    with open(job_path(job_id, "meta.json"), "w") as f:
        json.dump(meta, f)

    # プレビュー画像
    jpg, scale = render_check(img_lin, box, rod)
    with open(job_path(job_id, "preview.jpg"), "wb") as f:
        f.write(jpg)

    return {
        "job_id": job_id, "sample_id": sample_id, "source_kind": source_kind,
        "img_size": [img_lin.shape[1], img_lin.shape[0]],
        "preview_scale": scale,
        "auto_chart": box.tolist() if box is not None else None,
        "auto_rod": rod,
        "warnings": warnings,
    }


class ExtractReq(BaseModel):
    job_id: str
    chart_box: list          # 4x2 [[x,y],...]  左上→右上→右下→左下
    rod: dict                # {axis_y, top_y, bottom_y, x_start, x_end}
    save: bool = True


@app.post("/api/extract")
def extract(req: ExtractReq, x_api_key: str = Header(None)):
    auth(x_api_key)
    lin_path = job_path(req.job_id, "linear.npy")
    if not os.path.exists(lin_path):
        raise HTTPException(404, "ジョブが見つかりません (24時間で削除されます)。再アップロードしてください")
    img_lin = np.load(lin_path).astype(np.float64)
    with open(job_path(req.job_id, "meta.json")) as f:
        meta = json.load(f)
    sample_id = meta["sample_id"]
    warnings = []

    try:
        box = np.array(req.chart_box, dtype=np.float64)
        if box.shape != (4, 2):
            raise ValueError("chart_box は4点 [[x,y],...] で指定してください")
        M, white_norm, residual = compute_ccm(img_lin, box)
    except Exception as e:
        raise HTTPException(422, f"CCM計算失敗: {e}")

    if residual["mean_dE00"] > 4.0:
        warnings.append(diag("chart_low_quality", de=residual["mean_dE00"]))

    rod = {k: int(req.rod[k]) for k in ("axis_y", "top_y", "bottom_y", "x_start", "x_end")}
    if rod["bottom_y"] <= rod["top_y"] or rod["x_end"] <= rod["x_start"]:
        raise HTTPException(422, "棒領域の座標が不正です (top<bottom, x_start<x_end)")

    labs, band_warns = extract_cylinder(img_lin, rod, M, white_norm)
    warnings += band_warns

    # 確認画像更新
    jpg, _ = render_check(img_lin, box, rod)
    with open(job_path(req.job_id, "check.jpg"), "wb") as f:
        f.write(jpg)

    # sRGBスウォッチ (表示用)
    swatches = {}
    for a, lab in labs.items():
        if lab:
            XYZ = colour.Lab_to_XYZ(np.array(lab), illuminant=ILLUM_XY)
            rgb = np.clip(colour.XYZ_to_sRGB(XYZ, illuminant=ILLUM_XY), 0, 1)
            swatches[a] = '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in rgb)

    saved = False
    if req.save:
        try:
            hdr = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}",
                   "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
            # 既存チェック → upsert 的に: 同sample_nameの旧行を消して入れ直す
            rq.delete(f"{SUPA_URL}/rest/v1/iroca_calibration_shots",
                      params={"sample_name": f"eq.{sample_id}"}, headers=hdr, timeout=20)
            r = rq.post(f"{SUPA_URL}/rest/v1/iroca_calibration_shots", headers=hdr, timeout=30,
                        json={"sample_name": sample_id, "angles_deg": ANGLES,
                              "visual_labs": labs, "ccm_residual": residual,
                              "shot_metadata": {"source": meta["source_kind"], "job": req.job_id}})
            saved = r.status_code in (200, 201)
            if not saved:
                warnings.append(diag("supabase_error", detail=f"HTTP {r.status_code}"))
        except Exception as e:
            warnings.append(diag("supabase_error", detail=str(e)[:200]))

    return {"sample_id": sample_id, "ccm_residual": residual, "visual_labs": labs,
            "swatches": swatches, "saved": saved, "warnings": warnings}


@app.post("/api/learn")
def learn(x_api_key: str = Header(None)):
    auth(x_api_key)
    hdr = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}"}

    # 視覚データ
    r = rq.get(f"{SUPA_URL}/rest/v1/iroca_calibration_shots",
               params={"select": "sample_name,visual_labs", "order": "created_at.desc"},
               headers={**hdr, "Range": "0-999"}, timeout=30)
    r.raise_for_status()
    seen, visual = set(), {}
    for row in r.json():                      # 新しい順 → 同名は最新のみ
        if row["sample_name"] not in seen:
            visual[row["sample_name"]] = row["visual_labs"]
            seen.add(row["sample_name"])

    # i1Pro2 データ (ページング)
    i1, frm = {}, 0
    while True:
        r2 = rq.get(f"{SUPA_URL}/rest/v1/iroca_sample_summary",
                    params={"select": "sample_name,median_lab_l,median_lab_a,median_lab_b"},
                    headers={**hdr, "Range": f"{frm}-{frm+999}"}, timeout=30)
        batch = r2.json()
        for row in batch:
            if row["median_lab_l"] is not None:
                i1[row["sample_name"]] = [row["median_lab_l"], row["median_lab_a"], row["median_lab_b"]]
        if len(batch) < 1000:
            break
        frm += 1000

    warnings = []
    missing = [s for s in visual if s not in i1]
    if missing:
        warnings.append(diag("i1pro_missing", samples=missing))

    model = {"type": "affine_per_angle", "illuminant": "D50",
             "angles_deg": ANGLES, "per_angle": {}, "validation": {}}

    for a in ANGLES:
        X, Y, names = [], [], []
        for s, per in visual.items():
            if s in i1 and per.get(str(a)):
                X.append(i1[s]); Y.append(per[str(a)]); names.append(s)
        X, Y = np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)
        n = len(X)
        if n < 8:
            model["validation"][str(a)] = {"error": f"サンプル不足 ({n}件 < 8件)"}
            continue
        Xh = np.hstack([X, np.ones((n, 1))])
        W, *_ = np.linalg.lstsq(Xh, Y, rcond=None)
        A, b = W[:3].T, W[3]
        # LOOCV
        dEs = np.zeros(n)
        for i in range(n):
            m = np.arange(n) != i
            Wi, *_ = np.linalg.lstsq(Xh[m], Y[m], rcond=None)
            pred = Wi[:3].T @ X[i] + Wi[3]
            dEs[i] = delta_E_CIE2000(Y[i], pred)
        mean_cv, max_cv = float(dEs.mean()), float(dEs.max())
        worst = names[int(np.argmax(dEs))]
        if mean_cv > 5:
            warnings.append(diag("loocv_poor", angle=a, de=mean_cv, worst=worst))
        model["per_angle"][str(a)] = {"A": A.tolist(), "b": b.tolist(), "n_samples": n}
        model["validation"][str(a)] = {"loocv_mean_dE00": round(mean_cv, 2),
                                       "loocv_max_dE00": round(max_cv, 2),
                                       "loocv_worst_sample": worst}

    if not model["per_angle"]:
        n_total = len([s for s in visual if s in i1])
        raise HTTPException(422, detail=diag("insufficient_samples", n=n_total))

    # モデルを Supabase に保存 (旧active を無効化 → 新規insert)
    saved = False
    try:
        hdr2 = {**hdr, "Content-Type": "application/json"}
        rq.patch(f"{SUPA_URL}/rest/v1/iroca_correction_model",
                 params={"is_active": "eq.true"}, headers=hdr2,
                 json={"is_active": False}, timeout=20)
        r3 = rq.post(f"{SUPA_URL}/rest/v1/iroca_correction_model", headers=hdr2, timeout=30,
                     json={"model": model, "is_active": True})
        saved = r3.status_code in (200, 201)
        if not saved:
            warnings.append(diag("supabase_error", detail=f"モデル保存 HTTP {r3.status_code}"))
    except Exception as e:
        warnings.append(diag("supabase_error", detail=str(e)[:200]))

    return {"model": model, "saved": saved, "warnings": warnings,
            "n_visual_samples": len(visual), "n_matched": len([s for s in visual if s in i1])}


@app.get("/api/samples")
def samples(x_api_key: str = Header(None)):
    auth(x_api_key)
    hdr = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}"}
    r = rq.get(f"{SUPA_URL}/rest/v1/iroca_calibration_shots",
               params={"select": "sample_name,visual_labs,ccm_residual,created_at",
                       "order": "created_at.desc"},
               headers={**hdr, "Range": "0-999"}, timeout=30)
    r.raise_for_status()
    seen, out = set(), []
    for row in r.json():
        if row["sample_name"] not in seen:
            out.append(row)
            seen.add(row["sample_name"])
    return {"count": len(out), "samples": out}


@app.get("/api/image/{job_id}/{kind}")
def image(job_id: str, kind: str):
    if kind not in ("preview", "check"):
        raise HTTPException(404)
    p = job_path(job_id, f"{kind}.jpg")
    if not os.path.exists(p):
        raise HTTPException(404, "画像が見つかりません")
    with open(p, "rb") as f:
        return Response(f.read(), media_type="image/jpeg")


@app.exception_handler(Exception)
async def on_error(request, exc):
    return JSONResponse(status_code=500, content={
        "error": str(exc),
        "fix": "サーバーログを確認してください。繰り返す場合は入力画像の形式・サイズを変えて試してください。",
        "trace": traceback.format_exc()[-800:],
    })
