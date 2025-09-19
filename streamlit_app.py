# streamlit_app.py
# Taxonomy Guardian — Brand-Agnostic Brand Accuracy with NEW-suggestions, Match Strength,
# Family fallback, UPC assists, caching, and an Export NEW Suggestions button.
APP_VERSION = "TG 1.8 - Terra phrase BEFORE NEW extraction + compound token, stricter overlap, debug counters"

import io
import os
import re
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ======================================================
# Streamlit import with graceful fallback (shim for non-Streamlit environments)
# ======================================================
try:
    import streamlit as st  # type: ignore
except Exception:  # Provide a tiny shim so the module can import/run tests without Streamlit
    class _ExpanderShim:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    class _CtxShim:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    class _SidebarShim:
        def header(self, *a, **k): pass
        def markdown(self, *a, **k): pass
        def file_uploader(self, *a, **k): return None
        def selectbox(self, label, options=(), index=0, **k):
            try: return options[index]
            except Exception: return None
        def checkbox(self, label, value=False, **k): return value
        def number_input(self, *a, **k): return k.get("value", 0)
        def slider(self, *a, **k): return k.get("value", 0)
        def button(self, *a, **k): return False
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
    class _StShim:
        def __init__(self):
            self.session_state = {}
            self.sidebar = _SidebarShim()
            self.secrets = {}
        def set_page_config(self, *a, **k): pass
        def columns(self, spec): return _CtxShim(), _CtxShim()
        def image(self, *a, **k): pass
        def write(self, *a, **k): pass
        def title(self, *a, **k): pass
        def caption(self, *a, **k): pass
        def header(self, *a, **k): pass
        def subheader(self, *a, **k): pass
        def dataframe(self, *a, **k): pass
        def json(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def download_button(self, *a, **k): pass
        def button(self, *a, **k): return False
        def expander(self, *a, **k): return _ExpanderShim()
        def cache_data(self, *c, **ck):
            def deco(fn): return fn
            return deco
        def spinner(self, *a, **k): return _CtxShim()
    st = _StShim()  # type: ignore

# =========================
# App Config
# =========================
st.set_page_config(page_title="Taxonomy_Guardian", page_icon="🛡️", layout="wide")

# =========================
# Header
# =========================
try:
    c1, c2 = st.columns([0.15, 0.85])
    with c1:
        st.image("Taxonomy_Guardian.png", use_container_width=True)
    with c2:
        st.title("Taxonomy Guardian")
        st.caption("Ensuring the Fetch Taxonomy remains our source of truth! — " + APP_VERSION)
except Exception:
    st.title("Taxonomy Guardian")

# =========================
# Session logs
# =========================
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_event(level: str, msg: str, **data):
    st.session_state["logs"].append({"ts": _now(), "level": level, "msg": msg, **data})

def clear_logs():
    st.session_state["logs"] = []

# =========================
# Columns and constants
# =========================
REQUIRED_COLUMNS = [
    "FIDO","BARCODE","CATEGORY_HIERARCHY","CATEGORY_1","CATEGORY_2","CATEGORY_3","CATEGORY_4","DESCRIPTION","MANUFACTURER","BRAND","FIDO_TYPE",
]

OUTPUT_COLUMNS = [
    "Correct Brand?","Correct Categories?","Vague Description?","Suggested Brand",
    "Suggested Category 1","Suggested Category 2","Suggested Category 3","Suggested New Description",
    "Match Strength",
]

GENERIC_BRAND_TERMS = {"generic","unknown","n/a","na","misc","private label"}

# For filtering bad suggestions
_GENERIC_SINGLE_BRAND_TOKENS = {"garden","fitness","beauty","baby","wine","sports","snacks"}

# For NEW extraction stopwords
_NEW_STOPWORDS = set([
    "pack","size","flavor","assorted","variety","brand","product","item","mix","mixed",
    "cabernet","sauvignon","merlot","pinot","noir","chardonnay","red","white","rosé","rose",
    "sports","exercise","treadmill","bike","pro","max","ultra",
    "chips","crisps","nuts","trail","snack","snacks",
])

# =========================
# Basic utils
# =========================
def normalize_text(x: str) -> str:
    return str(x).strip().lower()

def normalized_header_map(cols: List[str]) -> Dict[str, str]:
    incoming_norm = {c: c.strip().upper() for c in cols}
    required_set = set(REQUIRED_COLUMNS)
    mapping: Dict[str, str] = {}
    used_targets = set()
    for raw, norm in incoming_norm.items():
        if norm in required_set and norm not in used_targets:
            mapping[raw] = norm
            used_targets.add(norm)
    return mapping

def coerce_required_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    mapping = normalized_header_map(df.columns.tolist())
    if mapping:
        df = df.rename(columns=mapping)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in ["Correct Brand?","Correct Categories?","Vague Description?"]:
        if col in df.columns:
            df[col] = df[col].replace({"": "N"})
    return df

# =========================
# Tokenization and fuzzy helpers
# =========================
def _tg_tokenize(text: str) -> List[str]:
    s = str(text or "").lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

def _tg_lev1(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 2
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
        if min(prev) > 1:
            return 2
    return prev[-1]

def _tg_vowel_eq(a: str, b: str) -> bool:
    strip = lambda x: re.sub(r"[aeiou]", "", x)
    return strip(a) == strip(b) and len(a) > 3 and len(b) > 3

def _tg_whole_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    patt = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(patt, str(text).lower()) is not None

def _normalize_for_phrase_scan(text: str) -> str:
    return re.sub(r"[\-/]+", " ", str(text))

def _title_case_brand(phrase: str) -> str:
    return " ".join(w.capitalize() for w in str(phrase).split())

# NEW helper: strip corporate suffixes for cleaner phrases

def _tg_strip_corp_suffix(phrase: str) -> str:
    suffixes = {"co","co.","inc","inc.","llc","l.l.c.","ltd","ltd.","corp","corp.","company"}
    parts = str(phrase).strip().split()
    while parts and parts[-1].lower() in suffixes:
        parts.pop()
    return " ".join(parts)

# =========================
# Family detection for fallback only
# =========================
def _tg_detect_family(text: str) -> Optional[str]:
    t = str(text or "").lower()
    if any(w in t for w in ["wine","merlot","cabernet","pinot","chardonnay"]):
        return "Wine"
    if any(w in t for w in ["saucer","planter","pot","garden","soil","hose"]):
        return "Garden Supplies"
    if any(w in t for w in ["treadmill","exercise","fitness","bike","elliptical"]):
        return "Fitness"
    if any(w in t for w in ["chips","crisps","trail mix","nuts"]):
        return "Snacks"
    if any(w in t for w in ["lotion","serum","beauty","cream"]):
        return "Beauty"
    if any(w in t for w in ["diaper","stroller","baby"]):
        return "Baby"
    return None

# =========================
# Master brand index and NEW extraction
# =========================
def _tg_master_brands(brand_df: Optional[pd.DataFrame]) -> set:
    if brand_df is None or "BRAND" not in brand_df.columns:
        return set()
    return set(brand_df["BRAND"].astype(str).str.strip().str.lower().tolist())

def _tg_brand_token_index(brand_df: Optional[pd.DataFrame]):
    idx = []
    if brand_df is None or "BRAND" not in brand_df.columns:
        return idx
    for b in brand_df["BRAND"].astype(str).str.strip().str.lower().unique():
        toks = set(_tg_tokenize(b))
        if toks:
            idx.append((b, toks))
    return idx

def _tg_map_to_master(raw_brand: str, master: set) -> Optional[str]:
    if not raw_brand:
        return None
    rb = re.sub(r"[^a-z0-9 ]+", " ", str(raw_brand).lower()).strip()
    if rb in master:
        return rb
    rb2 = re.sub(r"\s+", " ", rb)
    if rb2 in master:
        return rb2
    cands = [m for m in master if m[:1] == rb2[:1]]
    best, bestd = None, 2
    for m in cands:
        d = _tg_lev1(m, rb2)
        if d < bestd:
            best, bestd = m, d
            if d == 0:
                break
    if best and bestd <= 1:
        return best
    return None

def _tg_extract_new_brand(desc: str) -> Optional[str]:
    """
    Extract brand-like phrases from DESCRIPTION only:
      - "X by Y"  → Terra by Battat
      - root + Fitness  → Xterra Fitness
      - 2–4 token phrases → Bloem Terra, Terra Cotta Pasta
    Returns Title Case or None.
    """
    if not desc:
        return None
    s = _normalize_for_phrase_scan(desc)

    # X by Y
    m = re.search(r"\b([A-Za-z][\w]+(?:\s+[A-Za-z][\w]+){0,2})\s+by\s+([A-Za-z][\w]+)\b", s, flags=re.IGNORECASE)
    if m:
        cand = f"{m.group(1)} by {m.group(2)}".strip()
        return _title_case_brand(cand)

    # ROOT ... fitness
    m = re.search(r"\b([A-Za-z][A-Za-z0-9]{2,})\b[^\n\r]{0,80}?\bfitness\b", s, flags=re.IGNORECASE)
    if m:
        root = m.group(1)
        return _title_case_brand(f"{root} Fitness")

    # 2–4 tokens
    tokens = re.findall(r"\b([A-Za-z][A-Za-z0-9]+)\b", s)
    best = None
    best_len = 0
    for n in (4, 3, 2):
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i+n]
            if n == 2 and window[1].lower() == "fitness":
                return _title_case_brand(" ".join(window))
            filtered = [t for t in window if t.lower() not in _NEW_STOPWORDS]
            if not filtered:
                continue
            phrase = " ".join(window)
            if len(phrase) > best_len:
                best = phrase
                best_len = len(phrase)
        if best:
            break
    return _title_case_brand(best) if best else None

# =========================
# UPC lookups
# =========================
def _get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

class _TGUPC:
    def __init__(self):
        self.session_cache = {}
        self.disk_path = ".upc_cache.json"
        try:
            with open(self.disk_path, "r", encoding="utf-8") as f:
                self.disk_cache = json.load(f)
        except Exception:
            self.disk_cache = {}

    def _save(self):
        try:
            with open(self.disk_path, "w", encoding="utf-8") as f:
                json.dump(self.disk_cache, f)
        except Exception:
            pass

    def lookup(self, barcodes: list, off_cap=300, off_interval=0.15, upcdb_cap=100, upcdb_interval=0.5) -> dict:
        out = {}
        for b in barcodes:
            if not b:
                continue
            if b in self.session_cache:
                out[b] = self.session_cache[b]
            elif b in self.disk_cache:
                out[b] = self.disk_cache[b]
        remaining = [b for b in barcodes if b and b not in out]
        if remaining:
            remaining = self._off(remaining, out, off_cap, off_interval)
            remaining = self._upcdb(remaining, out, upcdb_cap, upcdb_interval)
        for k, v in out.items():
            self.session_cache[k] = v
            self.disk_cache[k] = v
        self._save()
        return out

    def _off(self, barcodes, out, cap, interval):
        import requests
        n = 0
        for b in barcodes:
            if n >= cap:
                break
            try:
                r = requests.get(f"https://world.openfoodfacts.org/api/v2/product/{b}.json", timeout=6)
                if r.status_code == 200:
                    js = r.json()
                    brand = None
                    try:
                        brand = (js.get("product", {}).get("brands_tags", []) or [None])[0]
                    except Exception:
                        brand = None
                    if brand:
                        out[b] = {"brand": brand}
                elif r.status_code in (429, 500, 502, 503):
                    time.sleep(0.75)
            except Exception:
                pass
            n += 1
            time.sleep(interval)
        return [b for b in barcodes if b not in out]

    def _upcdb(self, barcodes, out, cap, interval):
        import requests
        api_key = _get_secret("upcitemdb_api_key")
        if not api_key:
            return barcodes
        endpoint = _get_secret("upcitemdb_endpoint", "https://api.upcitemdb.com/prod/trial/lookup")
        headers = {"user_key": api_key, "key_type": "free" if "trial" in endpoint else "paid"}
        n = 0
        for b in barcodes:
            if n >= cap:
                break
            try:
                r = requests.get(endpoint, params={"upc": b}, headers=headers, timeout=6)
                if r.status_code == 200:
                    js = r.json()
                    brand = None
                    try:
                        if js.get("code") == "OK" and js.get("total", 0) > 0:
                            items = js.get("items", [])
                            if items:
                                brand = items[0].get("brand")
                    except Exception:
                        brand = None
                    if brand:
                        out[b] = {"brand": brand}
                elif r.status_code in (429, 500, 502, 503):
                    time.sleep(0.75)
            except Exception:
                pass
            n += 1
            time.sleep(interval)
        return [b for b in barcodes if b not in out]

# =========================
# Brand Accuracy
# =========================
def brand_accuracy_cleanup(
    df: pd.DataFrame,
    brand_df: Optional[pd.DataFrame],
    selected_mfr: str,
    selected_brand: str,
    log_changes_only: bool = True,
    max_logs: int = 200,
) -> pd.DataFrame:
    df = df.copy()
    df = ensure_output_columns(df)
    if "Match Strength" not in df.columns:
        df["Match Strength"] = ""

    sb = str(selected_brand or "").strip()
    sb_lower = sb.lower()

    # Allowed types for belonging
    allowed_types: List[str] = []
    if brand_df is not None and {"ALLOWED_PRODUCT_TYPES","BRAND"}.issubset(brand_df.columns):
        hits = brand_df[brand_df["BRAND"].astype(str).str.lower() == sb_lower]
        if not hits.empty:
            raw = str(hits.iloc[0].get("ALLOWED_PRODUCT_TYPES", "")).strip()
            if raw:
                allowed_types = sorted({t.strip().lower() for t in raw.split(",") if t.strip()})

    # Missing or generic brand mask
    brand_norm_lower = df["BRAND"].astype(str).str.strip().str.lower()
    nullish = {"", "none", "null", "nan", "n/a", "na"}
    mask_generic = brand_norm_lower.isin(nullish) | brand_norm_lower.isin(GENERIC_BRAND_TERMS) | df["BRAND"].isna()

    # DESCRIPTION only for suggestions
    desc_col = df["DESCRIPTION"].astype(str)

    def belongs_to_selected(text: str) -> bool:
        if allowed_types:
            for ph in allowed_types:
                toks = _tg_tokenize(text)
                first = toks[0] if toks else ""
                if _tg_whole_phrase(text, ph) or _tg_vowel_eq(ph, first):
                    return True
            return False
        else:
            return _tg_whole_phrase(text, sb_lower) or (sb_lower in text.lower())

    df["_belongs"] = desc_col.apply(belongs_to_selected)
    df.loc[:, "Correct Brand?"] = np.where(mask_generic | ~df["_belongs"], "N", "Y")

    master_brands = _tg_master_brands(brand_df)
    brand_index = _tg_brand_token_index(brand_df)

    upc_needed: List[Tuple[int, str]] = []
    suggestions: List[str] = []
    strengths: List[str] = []

    # debug counters
    c_new = c_phrase = c_family = c_exact = c_overlap = c_upc = c_unclear = 0

    for idx, row in df.iterrows():
        desc = str(row.get("DESCRIPTION", ""))
        desc_norm = _normalize_for_phrase_scan(desc)
        desc_lower = desc_norm.lower()

        if row["Correct Brand?"] == "Y":
            suggestions.append("")
            strengths.append("High")
            continue

        made = False  # did we emit a suggestion yet for this row?

        # 1) SelectedBrand plus phrase or family (FIRST!)
        # treat Terra compounds like "TerraCotta" as a brand mention too (use original-cased desc)
        brand_mention = _tg_whole_phrase(desc_lower, sb_lower) or bool(re.search(rf"\b{re.escape(sb)}[A-Z]", desc))
        if brand_mention and not made:
            scan = _normalize_for_phrase_scan(desc)
            m2 = re.search(
                rf"(?i)\b{re.escape(sb)}\s+([A-Za-z][A-Za-z0-9]+(?:\s+[A-Za-z][A-Za-z0-9]+){0,3})",
                scan,
            )
            if m2:
                raw_tail = m2.group(1)
                TAIL_DENY = {
                    "pack","size","adult","supplements","beauty","hair","skin","nails",
                    "case","belt","iphone","capsules","tablets","powder","pwd","mix",
                    "estate","cellars"
                }
                tail_tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", raw_tail) if t.lower() not in TAIL_DENY]
                if tail_tokens:
                    candidate = _tg_strip_corp_suffix(f"{sb} {' '.join(tail_tokens)}").strip()
                    cand_lower = candidate.lower()

                    if cand_lower != sb_lower:
                        mapped = _tg_map_to_master(candidate, master_brands)
                        if mapped and mapped.strip().lower() != sb_lower:
                            suggestions.append(mapped.title())
                            strengths.append("High")
                            c_phrase += 1
                            made = True
                        else:
                            suggestions.append(f"{_title_case_brand(candidate)} (NEW)")
                            strengths.append("Medium")
                            c_phrase += 1
                            made = True
            if not made:
                fam = _tg_detect_family(desc)
                if fam:
                    suggestions.append(f"{sb} {fam}")
                    strengths.append("Low")
                    c_family += 1
                    made = True

        # 2) NEW brand extraction — only if we didn't emit from phrase/family
        if not made:
            new_phrase = _tg_extract_new_brand(desc_norm)
            if new_phrase:
                mapped = _tg_map_to_master(new_phrase, master_brands)
                if mapped:
                    if mapped.strip().lower() != sb_lower:
                        suggestions.append(mapped.title())
                        strengths.append("High")
                        c_new += 1
                        made = True
                else:
                    suggestions.append(f"{new_phrase} (NEW)")
                    strengths.append("Medium")
                    c_new += 1
                    made = True

        # 3) Master exact phrase hit
        if not made:
            best_brand, best_len = None, 0
            for b in master_brands:
                if _tg_whole_phrase(desc_lower, b):
                    if b.strip().lower() == sb_lower:
                        continue
                    if len(b) > best_len:
                        best_brand, best_len = b, len(b)
            if best_brand:
                suggestions.append(best_brand.title())
                strengths.append("High")
                c_exact += 1
                made = True

        # 4) Master token overlap (stricter acceptance)
        if not made:
            desc_tokens = set(_tg_tokenize(desc_lower))
            best_b = None
            best_score = -999
            best_len2 = 0
            best_toks_len = 0
            for b, toks in brand_index:
                overlap = len(toks & desc_tokens)
                token_bonus = 1 if len(toks) > 1 else 0
                if b.strip().lower() == sb_lower:
                    continue
                generic_penalty = -2 if (len(toks) == 1 and next(iter(toks)) in _GENERIC_SINGLE_BRAND_TOKENS) else 0
                score = overlap + token_bonus + generic_penalty
                if score > best_score or (score == best_score and len(b) > best_len2):
                    best_b, best_score, best_len2, best_toks_len = b, score, len(b), len(toks)
            if best_b and (best_score >= 2 or (best_score == 1 and best_toks_len > 1)):
                suggestions.append(best_b.title())
                strengths.append("High" if best_score >= 3 else "Medium")
                c_overlap += 1
                made = True

        # 5) UPC assist
        if not made:
            bc = str(row.get("BARCODE", "")).strip()
            if bc and bc.isdigit() and len(bc) >= 8:
                upc_needed.append((idx, bc))
                suggestions.append("")
                strengths.append("Low")
                c_upc += 1
                made = True

        # 6) Family fallback last resort
        if not made:
            fam = _tg_detect_family(desc)
            if fam:
                suggestions.append(f"{selected_brand} {fam}")
                strengths.append("Low")
                c_family += 1
            else:
                suggestions.append("Unclear")
                strengths.append("Low")
                c_unclear += 1

    # UPC pass
    if upc_needed:
        prov = _TGUPC()
        uniq = sorted(set(bc for _, bc in upc_needed))
        results = prov.lookup(uniq, off_cap=300, off_interval=0.15, upcdb_cap=100, upcdb_interval=0.5)
        by_bc: Dict[str, List[int]] = {}
        for i, bc in upc_needed:
            by_bc.setdefault(bc, []).append(i)
        if results:
            for bc, info in results.items():
                raw = (info or {}).get("brand")
                mapped = _tg_map_to_master(raw, master_brands)
                label = (mapped.title() if mapped else (str(raw).title() if raw else None))
                if label:
                    for i in by_bc.get(bc, []):
                        if not suggestions[i]:
                            if str(label).strip().lower() == sb_lower and df.iloc[i]["Correct Brand?"] == "N":
                                continue
                            suggestions[i] = label if mapped else f"{label} (NEW)"
                            strengths[i] = "High"
        for i, s in enumerate(suggestions):
            if s == "":
                suggestions[i] = "Unclear"
                strengths[i] = "Low"
                c_unclear += 1

    # Final filters
    for i, s in enumerate(suggestions):
        if not s or s == "Unclear":
            continue
        base = s.replace(" (NEW)", "").strip()
        if base.lower() in _GENERIC_SINGLE_BRAND_TOKENS:
            suggestions[i] = "Unclear"
            strengths[i] = "Low"
            continue
        if base.lower() == sb_lower and df.iloc[i]["Correct Brand?"] == "N":
            suggestions[i] = "Unclear"
            strengths[i] = "Low"
            continue
        if base.lower() not in master_brands and not s.endswith("(NEW)"):
            suggestions[i] = f"{base} (NEW)"

    df.loc[:, "Suggested Brand"] = suggestions
    df.loc[:, "Match Strength"] = strengths

    # Ensure Y rows have blank suggestion
    df.loc[df["Correct Brand?"].eq("Y"), "Suggested Brand"] = ""

    # Logs — single compact line so you can verify the path mix without UI widgets
    log_event("INFO", "Brand path counters", counts={
        "NEW": int(c_new), "PHRASE": int(c_phrase), "FAMILY": int(c_family),
        "EXACT": int(c_exact), "OVERLAP": int(c_overlap), "UPC": int(c_upc), "UNCLEAR": int(c_unclear)
    })

    return df

# =========================
# Minimal stubs for other cleanups
# =========================
def category_hierarchy_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_output_columns(df)
    ok = df[["CATEGORY_1","CATEGORY_2","CATEGORY_3","CATEGORY_4"]].notna().all(axis=1)
    df.loc[:, "Correct Categories?"] = np.where(ok, df.get("Correct Categories?","Y"), "N")
    return df

def vague_description_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_output_columns(df)
    desc = df["DESCRIPTION"].astype(str).str.lower().fillna("")
    generic_hits = desc.str.contains(r"\b(assorted|variety pack|misc|pack of|item|product|brand|flavor|size|mixed|n/a)\b", regex=True)
    df.loc[:, "Vague Description?"] = np.where(generic_hits, "Y", df.get("Vague Description?","N"))
    return df

# =========================
# Cached I O
# =========================
@st.cache_data(show_spinner=False)
def _read_brand_mfr_xlsx(path_or_buffer) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(path_or_buffer)
        df = df.rename(columns={c: c.strip().upper() for c in df.columns})
        for c in ["BRAND","MANUFACTURER","WEBSITE","ALLOWED_PRODUCT_TYPES"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _get_top50_and_brand_lists(brand_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    if brand_df is None or brand_df.empty:
        return [], {}
    mf_series = brand_df["MANUFACTURER"].dropna().astype(str)
    top_50 = mf_series.value_counts().head(50).index.tolist()
    by_mfr: Dict[str, List[str]] = (
        brand_df.dropna(subset=["MANUFACTURER","BRAND"]).groupby("MANUFACTURER")["BRAND"].apply(lambda s: sorted(s.astype(str).unique())).to_dict()
    )
    return top_50, by_mfr

@st.cache_data(show_spinner=False)
def _read_user_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO, StringIO
    name = (file_name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))
    else:
        return pd.read_excel(BytesIO(file_bytes))

# =========================
# Sidebar controls
# =========================
def load_brand_mfr_reference(sidebar=True) -> Optional[pd.DataFrame]:
    brand_df = _read_brand_mfr_xlsx("All_Brands_Manufacturers.xlsx")
    if brand_df is None and sidebar:
        st.sidebar.markdown("##### Load Brand–Manufacturer Reference")
        uploaded_ref = st.sidebar.file_uploader("Upload All_Brands_Manufacturers.xlsx", type=["xlsx","xls"], key="brand_mfr_uploader")
        if uploaded_ref is not None:
            brand_df = _read_brand_mfr_xlsx(uploaded_ref)
            if brand_df is None:
                st.sidebar.error("Failed to read reference file.")
    return brand_df

def sidebar_controls(brand_df: Optional[pd.DataFrame]):
    st.sidebar.header("Inputs")
    uploaded = st.sidebar.file_uploader("Upload data file (from Snowflake export)", type=["xlsx","xls","csv"], key="data_upl")

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_key = f"{uploaded.name}:{hash(file_bytes)}"
        if st.session_state.get("_last_file_key") != file_key:
            st.session_state["_last_file_key"] = file_key
            st.session_state["_raw_df"] = _read_user_file(uploaded.name, file_bytes)
    raw_df = st.session_state.get("_raw_df")

    cleanup_choice = st.sidebar.selectbox("Type of cleanup", ["Brand Accuracy","Category Hierarchy Cleanup","Vague Description Cleanup"], index=0)

    selected_mfr = selected_brand = None
    if cleanup_choice == "Brand Accuracy":
        st.sidebar.markdown("#### Brand Accuracy Settings")
        show_top_only = st.sidebar.checkbox("Limit to Top 50 Manufacturers", value=True)
        manufacturer_options: List[str] = []
        brand_options: List[str] = []
        if brand_df is not None and not brand_df.empty:
            top_50, by_mfr = _get_top50_and_brand_lists(brand_df)
            manufacturer_options = top_50 if show_top_only else sorted(list(by_mfr.keys()))
            selected_mfr = st.sidebar.selectbox("Select Manufacturer", options=manufacturer_options)
            if selected_mfr:
                brand_options = by_mfr.get(selected_mfr, [])
            selected_brand = st.sidebar.selectbox("Select Brand", options=brand_options)
            st.sidebar.info(f"Selected: **{selected_brand}** (Manufacturer: **{selected_mfr}**) — {APP_VERSION}" )
        else:
            st.sidebar.warning("Load the Brand–Manufacturer reference to enable selections.")

    st.sidebar.markdown("#### Runtime Controls")
    log_changes_only = st.sidebar.checkbox("Log only changed or flagged rows", value=True)
    max_logs = st.sidebar.number_input("Max rows to log", min_value=50, max_value=2000, value=200, step=50)
    min_interval = st.sidebar.slider("Min interval between external calls (sec)", 0.0, 1.0, 0.15, 0.05)
    run = st.sidebar.button("Run Cleanup", use_container_width=True)

    return uploaded, raw_df, cleanup_choice, selected_mfr, selected_brand, log_changes_only, max_logs, min_interval, run

# =========================
# Results rendering after run only
# =========================
def show_results(cleaned: pd.DataFrame):
    st.subheader("Results")
    st.dataframe(cleaned.head(200), use_container_width=True)

    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="Cleaned")
    st.download_button(
        "Download cleaned file (.xlsx)",
        data=out_buf.getvalue(),
        file_name="taxonomy_guardian_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if "Suggested Brand" in cleaned.columns:
        mask_new = cleaned["Suggested Brand"].astype(str).str.endswith("(NEW)")
        new_rows = cleaned.loc[mask_new, [
            "FIDO","BARCODE","DESCRIPTION","BRAND","MANUFACTURER","Suggested Brand","Match Strength"
        ]].copy()
        if not new_rows.empty:
            buf2 = io.BytesIO()
            with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
                new_rows.to_excel(writer, index=False, sheet_name="NEW_Suggestions")
            st.download_button(
                "Export NEW Suggestions (.xlsx)",
                data=buf2.getvalue(),
                file_name="taxonomy_guardian_NEW_suggestions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# =========================
# Main
# =========================
def main():
    brand_df = load_brand_mfr_reference(sidebar=True)

    uploaded, raw_df, cleanup_choice, selected_mfr, selected_brand, log_changes_only, max_logs, min_interval, run = sidebar_controls(brand_df)

    st.subheader("Results")

    if uploaded is None or raw_df is None:
        st.info("Upload a data file to begin.")
        with st.expander("Logs", expanded=False):
            logs = st.session_state.get("logs", [])[-200:]
            st.json(logs)
        st.button("Clear logs", on_click=clear_logs)
        return

    if not run:
        st.caption("File loaded. Choose settings and click Run Cleanup to process.")
        with st.expander("Logs", expanded=False):
            logs = st.session_state.get("logs", [])[-200:]
            st.json(logs)
        st.button("Clear logs", on_click=clear_logs)
        return

    try:
        df = coerce_required_columns(raw_df)
    except Exception as e:
        st.error(f"Failed to prepare data: {e}")
        return

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns auto filled as empty: {', '.join(missing)}")

    cleaned = df.copy()

    if cleanup_choice == "Brand Accuracy":
        if not selected_mfr or not selected_brand:
            st.error("Please select both Manufacturer and Brand in the sidebar.")
            return
        with st.spinner("Running Brand Accuracy cleanup..."):
            cleaned = brand_accuracy_cleanup(
                cleaned,
                brand_df,
                selected_mfr=selected_mfr,
                selected_brand=selected_brand,
                log_changes_only=log_changes_only,
                max_logs=max_logs,
            )
    elif cleanup_choice == "Category Hierarchy Cleanup":
        with st.spinner("Running Category Hierarchy cleanup..."):
            cleaned = category_hierarchy_cleanup(cleaned)
    elif cleanup_choice == "Vague Description Cleanup":
        with st.spinner("Running Vague Description cleanup..."):
            cleaned = vague_description_cleanup(cleaned)

    show_results(cleaned)

    with st.expander("Logs", expanded=False):
        logs = st.session_state.get("logs", [])[-200:]
        st.json(logs)
    st.button("Clear logs", on_click=clear_logs)

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    main()
