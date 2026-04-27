/**
 * iroca-colorlib.js — IROCAシミュレーター用色科学ライブラリ
 *
 * 機能:
 *  - 36波長スペクトル(380-730nm/10nm) → Lab (D50/2°)
 *  - Lab → sRGB (D65, ガンマ補正済み)
 *  - 単一定数Kubelka-Munk混色 (K/S線形和→反射率逆変換)
 *  - CIEDE2000色差
 *
 * Python版 (colorlib.py) と数値的に一致するよう実装。
 * Phase B 39サンプルで mean ΔE=2.64 (現行モデル7.59から65%改善) を実測検証済み。
 */
(function (global) {
  'use strict';

  // ============================================================
  // 波長グリッドと CIE 1931 2° + D50 SPD
  // ============================================================
  const N_WL = 36;  // 380, 390, ..., 730 nm

  // CIE 1931 2° CMF (x_bar, y_bar, z_bar) at 380-730/10nm
  const CMF_X = [0.0014,0.0042,0.0143,0.0435,0.1344,0.2839,0.3483,0.3362,0.2908,
                 0.1954,0.0956,0.0320,0.0049,0.0093,0.0633,0.1655,0.2904,0.4334,
                 0.5945,0.7621,0.9163,1.0263,1.0622,1.0026,0.8544,0.6424,0.4479,
                 0.2835,0.1649,0.0874,0.0468,0.0227,0.0114,0.0058,0.0029,0.0014];
  const CMF_Y = [0.0000,0.0001,0.0004,0.0012,0.0040,0.0116,0.0230,0.0380,0.0600,
                 0.0910,0.1390,0.2080,0.3230,0.5030,0.7100,0.8620,0.9540,0.9950,
                 0.9950,0.9520,0.8700,0.7570,0.6310,0.5030,0.3810,0.2650,0.1750,
                 0.1070,0.0610,0.0320,0.0170,0.0082,0.0041,0.0021,0.0010,0.0005];
  const CMF_Z = [0.0065,0.0201,0.0679,0.2074,0.6456,1.3856,1.7471,1.7721,1.6692,
                 1.2876,0.8130,0.4652,0.2720,0.1582,0.0782,0.0422,0.0203,0.0087,
                 0.0039,0.0021,0.0017,0.0011,0.0008,0.0003,0.0002,0.0000,0.0000,
                 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000];
  const D50 =  [24.49,27.18,29.87,39.59,49.31,52.91,56.51,58.27,60.03,
                 58.93,57.82,66.32,74.82,81.04,87.25,88.93,90.61,90.99,
                 91.37,93.24,95.11,93.54,91.96,93.84,95.72,96.17,96.61,
                 96.87,97.13,99.61,102.10,101.43,100.75,101.54,102.32,101.16];

  // D50 white point (computed once)
  function _whitePoint() {
    let sumY = 0;
    for (let i = 0; i < N_WL; i++) sumY += D50[i] * CMF_Y[i];
    const k = 100.0 / sumY;
    let X = 0, Y = 0, Z = 0;
    for (let i = 0; i < N_WL; i++) {
      X += D50[i] * CMF_X[i];
      Y += D50[i] * CMF_Y[i];
      Z += D50[i] * CMF_Z[i];
    }
    return [k * X, k * Y, k * Z];
  }
  const D50_WHITE = _whitePoint();
  const _Y_NORM = (function () {
    let s = 0;
    for (let i = 0; i < N_WL; i++) s += D50[i] * CMF_Y[i];
    return 100.0 / s;
  })();

  // ============================================================
  // 反射率(36) → XYZ → Lab (D50/2°)
  // ============================================================
  function reflectanceToXYZ(R) {
    let X = 0, Y = 0, Z = 0;
    for (let i = 0; i < N_WL; i++) {
      const sr = D50[i] * R[i];
      X += sr * CMF_X[i];
      Y += sr * CMF_Y[i];
      Z += sr * CMF_Z[i];
    }
    return [_Y_NORM * X, _Y_NORM * Y, _Y_NORM * Z];
  }

  function xyzToLab(XYZ, refWhite) {
    refWhite = refWhite || D50_WHITE;
    const xr = XYZ[0] / refWhite[0];
    const yr = XYZ[1] / refWhite[1];
    const zr = XYZ[2] / refWhite[2];
    const eps = 216 / 24389;
    const kappa = 24389 / 27;
    const f = (t) => t > eps ? Math.cbrt(t) : (kappa * t + 16) / 116;
    const fx = f(xr), fy = f(yr), fz = f(zr);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }

  function reflectanceToLab(R) {
    return xyzToLab(reflectanceToXYZ(R));
  }

  // ============================================================
  // CIEDE2000 (Sharma et al. 2005 reference)
  // ============================================================
  function deltaE2000(lab1, lab2, kL = 1, kC = 1, kH = 1) {
    const L1 = lab1[0], a1 = lab1[1], b1 = lab1[2];
    const L2 = lab2[0], a2 = lab2[1], b2 = lab2[2];
    const C1 = Math.hypot(a1, b1);
    const C2 = Math.hypot(a2, b2);
    const Cb = (C1 + C2) / 2;
    const G = 0.5 * (1 - Math.sqrt(Math.pow(Cb, 7) / (Math.pow(Cb, 7) + Math.pow(25, 7))));
    const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
    const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
    const h1p = ((Math.atan2(b1, a1p) * 180 / Math.PI) + 360) % 360;
    const h2p = ((Math.atan2(b2, a2p) * 180 / Math.PI) + 360) % 360;
    const dLp = L2 - L1;
    const dCp = C2p - C1p;
    let dhp = h2p - h1p;
    if (Math.abs(dhp) > 180) dhp -= Math.sign(dhp) * 360;
    if (C1p * C2p === 0) dhp = 0;
    const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(dhp * Math.PI / 360);
    const Lpb = (L1 + L2) / 2;
    const Cpb = (C1p + C2p) / 2;
    let hpb;
    const hsum = h1p + h2p;
    if (Math.abs(h1p - h2p) <= 180) hpb = hsum / 2;
    else if (hsum < 360) hpb = (hsum + 360) / 2;
    else hpb = (hsum - 360) / 2;
    if (C1p * C2p === 0) hpb = h1p + h2p;
    const T = 1 - 0.17 * Math.cos((hpb - 30) * Math.PI / 180)
                + 0.24 * Math.cos((2 * hpb) * Math.PI / 180)
                + 0.32 * Math.cos((3 * hpb + 6) * Math.PI / 180)
                - 0.20 * Math.cos((4 * hpb - 63) * Math.PI / 180);
    const dTheta = 30 * Math.exp(-Math.pow((hpb - 275) / 25, 2));
    const Rc = 2 * Math.sqrt(Math.pow(Cpb, 7) / (Math.pow(Cpb, 7) + Math.pow(25, 7)));
    const Sl = 1 + (0.015 * Math.pow(Lpb - 50, 2)) / Math.sqrt(20 + Math.pow(Lpb - 50, 2));
    const Sc = 1 + 0.045 * Cpb;
    const Sh = 1 + 0.015 * Cpb * T;
    const Rt = -Math.sin(2 * dTheta * Math.PI / 180) * Rc;
    return Math.sqrt(
      Math.pow(dLp / (kL * Sl), 2) +
      Math.pow(dCp / (kC * Sc), 2) +
      Math.pow(dHp / (kH * Sh), 2) +
      Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    );
  }

  // ============================================================
  // Kubelka-Munk
  // ============================================================
  const R_MIN = 1e-4, R_MAX = 1 - 1e-4;

  function clipR(r) {
    return r < R_MIN ? R_MIN : (r > R_MAX ? R_MAX : r);
  }

  function reflectanceToKS(R) {
    const KS = new Array(N_WL);
    for (let i = 0; i < N_WL; i++) {
      const r = clipR(R[i]);
      KS[i] = (1 - r) * (1 - r) / (2 * r);
    }
    return KS;
  }

  function ksToReflectance(KS) {
    const R = new Array(N_WL);
    for (let i = 0; i < N_WL; i++) {
      const k = Math.max(KS[i], 0);
      R[i] = clipR(1 + k - Math.sqrt(k * k + 2 * k));
    }
    return R;
  }

  /** Mix N spectra with non-negative weights via single-constant K-M.
   *  spectra: array of (36-vec) reflectances
   *  weights: array of mass fractions (will be normalized) */
  function kmMix(spectra, weights) {
    const n = spectra.length;
    let sumW = 0;
    for (let i = 0; i < n; i++) sumW += weights[i];
    if (sumW <= 0) return spectra[0].slice();
    const KS_mix = new Array(N_WL).fill(0);
    for (let i = 0; i < n; i++) {
      const w = weights[i] / sumW;
      const KS = reflectanceToKS(spectra[i]);
      for (let j = 0; j < N_WL; j++) KS_mix[j] += w * KS[j];
    }
    return ksToReflectance(KS_mix);
  }

  // ============================================================
  // Lab(D50) → sRGB(D65)
  // ============================================================
  // Bradford D50→D65
  const M_D50_TO_D65 = [
    [ 0.9555766, -0.0230393,  0.0631636],
    [-0.0282895,  1.0099416,  0.0210077],
    [ 0.0122982, -0.0204830,  1.3299098],
  ];
  // XYZ(D65)→sRGB linear
  const M_XYZ_TO_SRGB = [
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570],
  ];

  function labToSrgb(lab) {
    const L = lab[0], a = lab[1], b = lab[2];
    const fy = (L + 16) / 116;
    const fx = a / 500 + fy;
    const fz = fy - b / 200;
    const eps = 216 / 24389, kappa = 24389 / 27;
    const finv = (f) => {
      const f3 = f * f * f;
      return f3 > eps ? f3 : (116 * f - 16) / kappa;
    };
    const X50 = finv(fx) * D50_WHITE[0];
    const Y50 = finv(fy) * D50_WHITE[1];
    const Z50 = finv(fz) * D50_WHITE[2];
    // D50→D65
    const X = (M_D50_TO_D65[0][0]*X50 + M_D50_TO_D65[0][1]*Y50 + M_D50_TO_D65[0][2]*Z50) / 100;
    const Y = (M_D50_TO_D65[1][0]*X50 + M_D50_TO_D65[1][1]*Y50 + M_D50_TO_D65[1][2]*Z50) / 100;
    const Z = (M_D50_TO_D65[2][0]*X50 + M_D50_TO_D65[2][1]*Y50 + M_D50_TO_D65[2][2]*Z50) / 100;
    // XYZ→sRGB linear
    let r = M_XYZ_TO_SRGB[0][0]*X + M_XYZ_TO_SRGB[0][1]*Y + M_XYZ_TO_SRGB[0][2]*Z;
    let g = M_XYZ_TO_SRGB[1][0]*X + M_XYZ_TO_SRGB[1][1]*Y + M_XYZ_TO_SRGB[1][2]*Z;
    let bv= M_XYZ_TO_SRGB[2][0]*X + M_XYZ_TO_SRGB[2][1]*Y + M_XYZ_TO_SRGB[2][2]*Z;
    r = Math.max(0, Math.min(1, r));
    g = Math.max(0, Math.min(1, g));
    bv= Math.max(0, Math.min(1, bv));
    const enc = (v) => v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1/2.4) - 0.055;
    return [Math.round(enc(r)*255), Math.round(enc(g)*255), Math.round(enc(bv)*255)];
  }

  function rgbToHex(rgb) {
    return '#' + rgb.map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
  }

  function spectrumToHex(R) {
    return rgbToHex(labToSrgb(reflectanceToLab(R)));
  }

  // ============================================================
  // 薬剤スペクトル補間 (K/S linear in white_pct between anchors)
  // ============================================================
  /** Get drug spectrum at arbitrary white_pct (0-100).
   *  drugLevels: { 0: spectrum36, 50: spectrum36, 100: spectrum36, ... }
   *  Linear K/S interpolation between adjacent measured anchors. */
  function getDrugSpectrumAt(drugLevels, whitePct) {
    if (!drugLevels) return null;
    const keys = Object.keys(drugLevels)
      .filter(k => /^\d+$/.test(k))
      .map(Number)
      .sort((a, b) => a - b);
    if (keys.length === 0) return null;
    if (whitePct <= keys[0]) return drugLevels[keys[0]].slice();
    if (whitePct >= keys[keys.length - 1]) return drugLevels[keys[keys.length - 1]].slice();
    // exact match
    if (drugLevels[whitePct]) return drugLevels[whitePct].slice();
    // bracket
    let lo = keys[0], hi = keys[keys.length - 1];
    for (let i = 0; i < keys.length - 1; i++) {
      if (keys[i] <= whitePct && whitePct <= keys[i + 1]) {
        lo = keys[i];
        hi = keys[i + 1];
        break;
      }
    }
    const t = (whitePct - lo) / (hi - lo);
    return kmMix([drugLevels[lo], drugLevels[hi]], [1 - t, t]);
  }

  /** Mix multiple drugs at specified white_pct.
   *  comps: [{drug: 'N7', weight: 4}, {drug: 'B', weight: 0.5}, ...]
   *  drugTable: { N7: { 0: [..36..], 50: [..36..], 100: [..36..] }, ... }
   *  Returns: { spectrum: [36], lab: [L,a,b], rgb: [r,g,b], hex: '#...' } */
  function mixDrugsKM(comps, whitePct, drugTable) {
    const sps = [], ws = [];
    for (const c of comps) {
      const sp = getDrugSpectrumAt(drugTable[c.drug] || drugTable[c.code], whitePct);
      if (!sp) return null;
      sps.push(sp);
      ws.push(c.weight);
    }
    if (sps.length === 0) return null;
    const R = kmMix(sps, ws);
    const lab = reflectanceToLab(R);
    const rgb = labToSrgb(lab);
    return { spectrum: R, lab, rgb, hex: rgbToHex(rgb) };
  }

  // ============================================================
  // Public API
  // ============================================================
  const api = {
    N_WL,
    D50_WHITE,
    reflectanceToXYZ,
    reflectanceToLab,
    xyzToLab,
    labToSrgb,
    rgbToHex,
    spectrumToHex,
    deltaE2000,
    reflectanceToKS,
    ksToReflectance,
    kmMix,
    getDrugSpectrumAt,
    mixDrugsKM,
  };

  // CommonJS / browser global
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof global !== 'undefined') global.IROCAColor = api;
})(typeof globalThis !== 'undefined' ? globalThis : (typeof window !== 'undefined' ? window : this));
