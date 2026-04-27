/**
 * iroca-calibration.js — 実測スペクトルデータをSupabaseから動的に取得し、
 * シミュレーターの混色計算を Kubelka-Munk モデルへ切り替える。
 *
 * 動作:
 *  1. ページロード時に localStorage キャッシュ (TTL 24h) があれば即時利用
 *  2. バックグラウンドで Supabase の最新データを取得
 *  3. 取得成功 → DRUG_TABLE 更新 → window.IROCA_CAL_READY イベント発火
 *  4. 取得失敗 → キャッシュまたは画像推定値(BC[])にフォールバック
 *
 * 検証済み: Phase B 39サンプルで mean ΔE=2.64 (現行7.59から65%改善)
 */
(function (global) {
  'use strict';

  const SUPA_URL = 'https://flmeolcfutuwwbjmzyoz.supabase.co';
  const SUPA_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZsbWVvbGNmdXR1d3diam16eW96Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM5NzAxODYsImV4cCI6MjA3OTU0NjE4Nn0.VVxUxKexNeN6dUiAMDkCNlnIoXa-F5rfBqHPBDcwdnU';
  const CACHE_KEY = 'iroca_drug_table_v1';
  const CACHE_TTL_MS = 24 * 60 * 60 * 1000;  // 24 hours

  const HAIR_MAP = {
    '白髪0%（黒髪100%）': 0,
    '黒髪100%': 0,
    '白髪30%': 30,
    '白髪50%': 50,
    '白髪70%': 70,
    '白髪80%': 80,
    '白髪100%': 100
  };

  // Public state
  global.DRUG_TABLE = null;       // { drug: { hair_pct: spectrum[36], ... }, ... }
  global.CAL_META = null;         // { drugCount, sampleCount, lastUpdated, source }

  // ---------------------------------------------------------------
  // Fetch helpers
  // ---------------------------------------------------------------
  async function supaFetch(table, params, retries = 3) {
    const url = `${SUPA_URL}/rest/v1/${table}?${params}`;
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const r = await fetch(url, {
          headers: { apikey: SUPA_KEY, Authorization: `Bearer ${SUPA_KEY}` }
        });
        if (r.ok) return r.json();
        // 5xx or 429: retry; others: bail
        if (r.status >= 500 || r.status === 429) {
          lastErr = new Error(`Supabase ${r.status}`);
        } else {
          throw new Error(`Supabase ${r.status}: ${await r.text()}`);
        }
      } catch (e) {
        lastErr = e;
      }
      if (attempt < retries) {
        const wait = Math.min(8000, 1000 * Math.pow(2, attempt));  // 1s, 2s, 4s, 8s
        await new Promise(res => setTimeout(res, wait));
      }
    }
    throw lastErr;
  }

  function loadCache() {
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj || !obj.savedAt || !obj.drugTable) return null;
      if (Date.now() - obj.savedAt > CACHE_TTL_MS) return null;
      return obj;
    } catch (e) {
      return null;
    }
  }

  function saveCache(drugTable, meta) {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({
        savedAt: Date.now(), drugTable, meta
      }));
    } catch (e) {
      console.warn('[iroca-cal] localStorage save failed:', e);
    }
  }

  // ---------------------------------------------------------------
  // Build drug table from plan + summary
  // ---------------------------------------------------------------
  async function fetchAndBuild() {
    // Phase A 単品 plan rows
    // NOTE: Avoid URL-encoded Japanese (e.g. ratio=eq.単品) in PostgREST query —
    // it triggers "DNS cache overflow" 503 errors. Filter client-side instead.
    const plan = await supaFetch(
      'iroca_experiment_plan',
      'select=id,drug,ratio,hair_type,status&phase=eq.A&status=eq.measured'
    );
    const planById = {};
    const wanted = new Set();
    for (const p of plan) {
      if (p.ratio !== '単品') continue;  // client-side filter
      planById[p.id] = p;
      wanted.add(p.id);
    }
    // Sample summary rows (only those we want)
    const ids = Array.from(wanted);
    const summary = await supaFetch(
      'iroca_sample_summary',
      `select=sample_name,median_spectrum,measurement_count,updated_at&sample_name=in.(${ids.join(',')})`
    );

    const drugTable = {};
    let totalSamples = 0;
    let latest = 0;
    for (const s of summary) {
      const p = planById[s.sample_name];
      if (!p) continue;
      const drug = p.drug;
      const hp = HAIR_MAP[p.hair_type];
      if (drug == null || hp == null) continue;
      if (!s.median_spectrum || s.median_spectrum.length !== 36) continue;
      drugTable[drug] = drugTable[drug] || {};
      drugTable[drug][hp] = s.median_spectrum;
      totalSamples++;
      const t = new Date(s.updated_at).getTime();
      if (t > latest) latest = t;
    }
    const meta = {
      drugCount: Object.keys(drugTable).length,
      sampleCount: totalSamples,
      lastUpdated: latest ? new Date(latest).toISOString() : null,
      source: 'supabase',
      fetchedAt: new Date().toISOString()
    };
    return { drugTable, meta };
  }

  // ---------------------------------------------------------------
  // Initialize: cache-first, then live refresh
  // ---------------------------------------------------------------
  async function init() {
    const cached = loadCache();
    if (cached) {
      global.DRUG_TABLE = cached.drugTable;
      global.CAL_META = Object.assign({}, cached.meta, { source: 'cache' });
      dispatch('iroca:cal-ready', { fromCache: true });
    }
    try {
      const { drugTable, meta } = await fetchAndBuild();
      global.DRUG_TABLE = drugTable;
      global.CAL_META = meta;
      saveCache(drugTable, meta);
      dispatch('iroca:cal-ready', { fromCache: false, fresh: true });
    } catch (e) {
      console.warn('[iroca-cal] live fetch failed, using cache or fallback:', e);
      if (!global.DRUG_TABLE) {
        global.CAL_META = { source: 'fallback', error: String(e) };
        dispatch('iroca:cal-ready', { fromCache: false, fallback: true });
      }
    }
  }

  function dispatch(name, detail) {
    try {
      window.dispatchEvent(new CustomEvent(name, { detail }));
    } catch (e) { /* noop */ }
  }

  // ---------------------------------------------------------------
  // Public mix function — K-M when fully calibrated, else null (caller falls back)
  // ---------------------------------------------------------------
  /**
   * Spectral K-M mix. Returns null if any drug is not fully calibrated.
   * comps: [{code: 'N7', weight: 4}, ...]
   * whitePct: 0..100
   */
  function mixSpectralKM(comps, whitePct) {
    if (!global.DRUG_TABLE || !global.IROCAColor) return null;
    const C = global.IROCAColor;
    const sps = [], ws = [];
    for (const c of comps) {
      const levels = global.DRUG_TABLE[c.code];
      if (!levels || Object.keys(levels).length === 0) return null;
      const sp = C.getDrugSpectrumAt(levels, whitePct);
      if (!sp) return null;
      sps.push(sp);
      ws.push(c.weight);
    }
    if (sps.length === 0) return null;
    const R = C.kmMix(sps, ws);
    const lab = C.reflectanceToLab(R);
    const rgb = C.labToSrgb(lab);
    return {
      r: rgb[0], g: rgb[1], b: rgb[2],
      hex: C.rgbToHex(rgb),
      lab, spectrum: R,
      method: 'spectral-km'
    };
  }

  // ---------------------------------------------------------------
  // Auto-start
  // ---------------------------------------------------------------
  global.IROCACalibration = { init, mixSpectralKM, fetchAndBuild };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(typeof window !== 'undefined' ? window : globalThis);
