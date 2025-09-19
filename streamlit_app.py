# streamlit_app.py
# Taxonomy Guardian — Brand Accuracy with master mapping, cleaned phrases, heuristics,
# UPC safety-net (OpenFoodFacts + optional UPCItemDB), and "(UPC Lookup)" strength label.
# The sidebar "Selected: ..." info box has been removed per request.

import io
import re
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# =========================
# Streamlit import with graceful fallback (to run outside Streamlit)
# =========================
try:
    import streamlit as st  # type: ignore
except Exception:
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, a,b,c): return False
    class _SB:
        def header(self,*a,**k): pass
        def markdown(self,*a,**k): pass
        def file_uploader(self,*a,**k): return None
        def selectbox(self,*a,**k): return None
        def checkbox(self,*a,**k): return False
        def number_input(self,*a,**k): return 200
        def slider(self,*a,**k): return 0.15
        def button(self,*a,**k): return False
        def info(self,*a,**k): pass
        def warning(self,*a,**k): pass
        def error(self,*a,**k): pass
    class _St:
        def __init__(self):
            self.sidebar = _SB()
            self.session_state = {}
            self.secrets = {}
        def set_page_config(self, *a, **k): pass
        def image(self,*a,**k): pass
        def title(self,*a,**k): pass
        def caption(self,*a,**k): pass
        def header(self,*a,**k): pass
        def subheader(self,*a,**k): pass
        def dataframe(self,*a,**k): pass
        def json(self,*a,**k): pass
        def info(self,*a,**k): pass
        def warning(self,*a,**k): pass
        def error(self,*a,**k): pass
        def download_button(self,*a,**k): pass
        def expander(self,*a,**k): return _Ctx()
        def cache_data(self, *c, **ck):
            def deco(fn): return fn
            return deco
        def spinner(self, *a, **k): return _Ctx()
        def columns(self, *a, **k): return _Ctx(), _Ctx()
    st = _St()  # type: ignore

# =========================
# App config / header
# =========================
st.set_page_config(page_title="Taxonomy Guardian", page_icon="🛡️", layout="wide")

try:
    c1, c2 = st.columns([0.18, 0.82])
    with c1:
        st.image("Taxonomy_Guardian.png", use_container_width=True)
    with c2:
        st.title("Taxonomy Guardian")
        st.caption("Ensuring the Fetch Taxonomy remains our source of truth")
except Exception:
    st.title("Taxonomy Guardian")

# =========================
# Logs
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
# Columns / constants
# =========================
REQUIRED_COLUMNS = [
    "FIDO","BARCODE","CATEGORY_HIERARCHY","CATEGORY_1","CATEGORY_2","CATEGORY_3","CATEGORY_4",
    "DESCRIPTION","MANUFACTURER","BRAND","FIDO_TYPE",
]

OUTPUT_COLUMNS = [
    "Correct Brand?","Correct Categories?","Vague Description?","Suggested Brand",
    "Suggested Category 1","Suggested Category 2","Suggested Category 3","Suggested New Description",
    "Match Strength",
]

GENERIC_BRAND_TERMS = {"generic","unknown","n/a","na","misc","private label"}

_UNITS = {
    "ml","oz","fl","floz","ct","pack","pk","pk.","3pk","6pk","12pk","case","g","kg","lb","lbs"
}
_FORM_WORDS = {
    "frozen","roasted","garlic","powder","capsules","tablet","tablets","softgels","gummy","gummies",
    "ravioli","sauce","mix","assorted","variety","single","multi","bottle","bottles","can","cans",
    "pods","k-cup","kcup","kcups","keurig","ground","whole","bean","beans"
}
_DECOR_NOISE = {
    "halloween","witch","party","led","lights","mini","battery","powered","outdoor","yard","garden",
    "tree","black","red","purple"
}

GENERIC_SINGLE_BRAND_TOKENS = {"garden","fitness","beauty","baby","wine","sports","snacks"}

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
# Tokenization / fuzzy helpers
# =========================
def _tg_tokenize(text: str) -> List[str]:
    s = str(text or "").lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

def _tg_lev1(a: str, b: str) -> int:
    if a == b: return 0
    if abs(len(a)-len(b)) > 1: return 2
    m, n = len(a), len(b)
    if m > n: a, b, m, n = b, a, n, m
    prev = list(range(n+1))
    for i in range(1, m+1):
        cur = [i] + [0]*n
        ai = a[i-1]
        for j in range(1, n+1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
        prev = cur
        if min(prev) > 1: return 2
    return prev[-1]

def _tg_vowel_eq(a: str, b: str) -> bool:
    strip = lambda x: re.sub(r"[aeiou]", "", x)
    return strip(a) == strip(b) and len(a)>3 and len(b)>3

def _tg_whole_phrase(text: str, phrase: str) -> bool:
    if not phrase: return False
    patt = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(patt, str(text).lower()) is not None

def _normalize_for_phrase_scan(text: str) -> str:
    return re.sub(r"[\-/]+", " ", str(text))

def _title_case_brand(phrase: str) -> str:
    return " ".join(w.capitalize() for w in str(phrase).split())

def _tg_strip_corp_suffix(phrase: str) -> str:
    suffixes = {"co","co.","inc","inc.","llc","l.l.c.","ltd","ltd.","corp","corp.","company"}
    parts = str(phrase).strip().split()
    while parts and parts[-1].lower() in suffixes:
        parts.pop()
    return " ".join(parts)

# =========================
# Phrase cleaner (caps to 2–3 tokens, strips units/forms/decor noise)
# =========================
def _tg_clean_phrase(core: str) -> str:
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", core)]
    cleaned = []
    for t in tokens:
        tl = t.lower()
        if tl in _UNITS or tl in _FORM_WORDS or tl in _DECOR_NOISE:
            continue
        cleaned.append(t)
        if len(cleaned) >= 3:
            break
    if not cleaned:
        return ""
    out = " ".join(cleaned)
    out = _tg_strip_corp_suffix(out)
    return _title_case_brand(out)

# =========================
# Family (heuristic) detector — low confidence fallback
# =========================
def _tg_detect_family(text: str) -> Optional[str]:
    t = str(text or "").lower()
    if any(w in t for w in ["wine","merlot","cabernet","pinot","chardonnay","sauvignon","syrah","rioja","tempranillo"]):
        return "Wine"
    if any(w in t for w in ["decor","decoration","halloween","witch","ornament","garland","lights"]):
        return "Decor"
    if any(w in t for w in ["capsule","capsules","powder","gummy","gummies","supplement","supplements"]):
        return "Supplements"
    if any(w in t for w in ["tile","marble","terrazzo","ceramic","porcelain","granite","sample"]):
        return "Tile"
    if any(w in t for w in ["fitness","treadmill","exercise","bike","belt"]):
        return "Fitness"
    if any(w in t for w in ["coffee","keurig","k-cup","kcup","kcups","ground","espresso"]):
        return "Coffee"
    return None

# =========================
# Master brand index + NEW extraction
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

def _tg_map_to_master(raw_brand: Optional[str], master: set) -> Optional[str]:
    if not raw_brand:
        return None
    rb = re.sub(r"[^a-z0-9 ]+", " ", str(raw_brand).lower()).strip()
    if not rb:
        return None
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

    # 2–3 tokens (conservative)
    tokens = re.findall(r"\b([A-Za-z][A-Za-z0-9]+)\b", s)
    best = None
    best_len = 0
    for n in (3, 2):
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i+n]
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
# UPC lookups (OFF first; then UPCItemDB if available in secrets)
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
                        out[b] = {"source":"OFF","brand": brand}
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
                        out[b] = {"source":"UPCItemDB","brand": brand}
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

    # Allowed product types from master (for selected brand only)
    allowed_types: List[str] = []
    if brand_df is not None and {"ALLOWED_PRODUCT_TYPES","BRAND"}.issubset(brand_df.columns):
        hits = brand_df[brand_df["BRAND"].astype(str).str.lower() == sb_lower]
        if not hits.empty:
            raw = str(hits.iloc[0].get("ALLOWED_PRODUCT_TYPES", "")).strip()
            if raw:
                allowed_types = sorted({t.strip().lower() for t in raw.split(",") if t.strip()})

    brand_norm_lower = df["BRAND"].astype(str).str.strip().str.lower()
    nullish = {"", "none", "null", "nan", "n/a", "na"}
    mask_generic = brand_norm_lower.isin(nullish) | brand_norm_lower.isin(GENERIC_BRAND_TERMS) | df["BRAND"].isna()

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

    suggestions: List[str] = []
    strengths: List[str] = []
    upc_needed: List[Tuple[int, str]] = []

    for idx, row in df.iterrows():
        desc = str(row.get("DESCRIPTION", ""))
        desc_norm = _normalize_for_phrase_scan(desc)
        desc_lower = desc_norm.lower()

        if row["Correct Brand?"] == "Y":
            suggestions.append("")
            strengths.append("High")
            continue

        # ---- 1) Master mapping from description (exact/near) ----
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
            continue

        # ---- 2) SelectedBrand + cleaned phrase (up to 3 tokens) ----
        if _tg_whole_phrase(desc_lower, sb_lower):
            m2 = re.search(
                rf"\b{re.escape(sb)}\s+([A-Za-z][A-Za-z0-9]+(?:\s+[A-Za-z][A-Za-z0-9]+){{0,3}})",
                desc_norm,
                flags=re.IGNORECASE,
            )
            if m2:
                raw_tail = m2.group(1)
                cleaned = _tg_clean_phrase(f"{sb} {raw_tail}")
                if cleaned and cleaned.lower() != sb_lower:
                    mapped = _tg_map_to_master(cleaned, master_brands)
                    if mapped and mapped.strip().lower() != sb_lower:
                        suggestions.append(mapped.title())
                        strengths.append("High")
                        continue
                    else:
                        suggestions.append(f"{cleaned} (NEW)")
                        strengths.append("Medium")
                        continue

        # ---- 3) Heuristic fallback (low confidence) ----
        fam = _tg_detect_family(desc)
        if fam:
            suggestions.append(f"{selected_brand} {fam}")
            strengths.append("Low")
        else:
            suggestions.append("")
            strengths.append("")

        # ---- 4) UPC lookup (safety net) ----
        bc = str(row.get("BARCODE", "")).strip()
        need_upc = (bc and bc.isdigit() and len(bc) >= 8)
        already_have = bool(suggestions[-1]) and suggestions[-1] not in ("", "Unclear")
        if need_upc and not already_have:
            upc_needed.append((idx, bc))

    # UPC pass (batch)
    if upc_needed:
        prov = _TGUPC()
        uniq = sorted(set(bc for _, bc in upc_needed))
        results = prov.lookup(uniq, off_cap=300, off_interval=0.15, upcdb_cap=100, upcdb_interval=0.5)
        by_bc: Dict[str, List[int]] = {}
        for i, bc in upc_needed:
            by_bc.setdefault(bc, []).append(i)

        for bc, info in results.items():
            raw = (info or {}).get("brand")
            mapped = _tg_map_to_master(raw, master_brands)
            label = (mapped.title() if mapped else (_title_case_brand(str(raw)) if raw else None))
            for i in by_bc.get(bc, []):
                if not label:
                    continue
                if not suggestions[i] or strengths[i] in ("", "Low"):
                    suggestions[i] = label if mapped else f"{label} (NEW)"
                    base_strength = "High" if mapped else "Medium"
                    strengths[i] = f"{base_strength} (UPC Lookup)"

    # Fill remaining rows (NEW / Unclear)
    for i in range(len(suggestions)):
        if suggestions[i]:
            continue
        desc = str(df.iloc[i].get("DESCRIPTION", ""))
        new_phrase = _tg_extract_new_brand(_normalize_for_phrase_scan(desc))
        if new_phrase:
            mapped = _tg_map_to_master(new_phrase, master_brands)
            if mapped and mapped.strip().lower() != sb_lower:
                suggestions[i] = mapped.title()
                strengths[i] = "Medium"
                continue
            elif new_phrase.lower() != sb_lower:
                suggestions[i] = f"{new_phrase} (NEW)"
                strengths[i] = "Medium"
                continue
        suggestions[i] = "Unclear"
        strengths[i] = "Low"

    # Final filters: block generic single tokens and base brand on N rows
    for i, s in enumerate(suggestions):
        if not s or s == "Unclear":
            continue
        base = s.replace(" (NEW)", "").strip()
        if base.lower() in GENERIC_SINGLE_BRAND_TOKENS:
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
    df.loc[df["Correct Brand?"].eq("Y"), "Match Strength"] = "High"

    # Logs
    changed = df[df["Correct Brand?"].eq("N")]
    for _, r in changed.head(max_logs).iterrows() if log_changes_only else changed.iterrows():
        log_event(
            "INFO", "Brand check (final)",
            fido=str(r.get("FIDO")),
            brand=str(r.get("BRAND")),
            desc=str(r.get("DESCRIPTION"))[:200],
            suggested=str(r.get("Suggested Brand")),
            strength=str(r.get("Match Strength")),
        )
    return df

# =========================
# Other cleanups
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
# Cached I/O
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
        brand_df.dropna(subset=["MANUFACTURER","BRAND"]).groupby("MANUFACTURER")["BRAND"]
        .apply(lambda s: sorted(s.astype(str).unique())).to_dict()
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
# Sidebar controls (info box removed)
# =========================
def sidebar_controls(brand_df: Optional[pd.DataFrame]):
    st.sidebar.header("Inputs")
    uploaded = st.sidebar.file_uploader(
        "Upload data file (from Snowflake export)",
        type=["xlsx","xls","csv"],
        key="data_upl"
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_key = f"{uploaded.name}:{hash(file_bytes)}"
        if st.session_state.get("_last_file_key") != file_key:
            st.session_state["_last_file_key"] = file_key
            st.session_state["_raw_df"] = _read_user_file(uploaded.name, file_bytes)
    raw_df = st.session_state.get("_raw_df")

    cleanup_choice = st.sidebar.selectbox(
        "Type of cleanup",
        ["Brand Accuracy","Category Hierarchy Cleanup","Vague Description Cleanup"],
        index=0
    )

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
        else:
            st.sidebar.warning("Load the Brand–Manufacturer reference to enable selections.")

    st.sidebar.markdown("#### Runtime Controls")
    log_changes_only = st.sidebar.checkbox("Log only changed/flagged rows", value=True)
    max_logs = st.sidebar.number_input("Max rows to log", min_value=50, max_value=2000, value=200, step=50)
    min_interval = st.sidebar.slider("Min interval between external calls (sec)", 0.0, 1.0, 0.15, 0.05)
    run = st.sidebar.button("Run Cleanup", use_container_width=True)

    return uploaded, raw_df, cleanup_choice, selected_mfr, selected_brand, log_changes_only, max_logs, min_interval, run

# =========================
# Results rendering
# =========================
def show_results(cleaned: pd.DataFrame):
    st.subheader("Results")
    st.dataframe(cleaned.head(300), use_container_width=True)

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

    st.subheader("Run")

    if uploaded is None or raw_df is None:
        st.info("Upload a data file to begin.")
        with st.expander("Logs", expanded=False):
            logs = st.session_state.get("logs", [])[-200:]
            st.json(logs)
        st.button("Clear logs", on_click=clear_logs)
        return

    if not run:
        st.caption("File loaded. Choose settings and click **Run Cleanup** to process.")
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
        st.warning(f"Missing required columns (auto-filled as empty): {', '.join(missing)}")

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

if __name__ == "__main__":
    main()
