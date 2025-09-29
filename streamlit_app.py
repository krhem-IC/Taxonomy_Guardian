# streamlit_app.py
# Taxonomy Guardian v12 - Complete syntax-validated version

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
# Streamlit import with graceful fallback
# =========================
try:
    import streamlit as st
except ImportError:
    class MockSt:
        def __init__(self):
            self.sidebar = self
            self.session_state = {}
            self.secrets = {}
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = MockSt()

# =========================
# App config
# =========================
st.set_page_config(page_title="Taxonomy Guardian", page_icon="🛡️", layout="wide")

try:
    c1, c2 = st.columns([0.18, 0.82])
    with c1:
        try:
            st.image("Taxonomy_Guardian.png", use_container_width=True)
        except:
            st.markdown("🛡️")
    with c2:
        st.title("Taxonomy Guardian")
        st.caption("Ensuring the Fetch Taxonomy remains our source of truth")
except Exception:
    st.title("Taxonomy Guardian")
    st.caption("Ensuring the Fetch Taxonomy remains our source of truth")

# =========================
# Logging System
# =========================
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_event(level: str, msg: str, **data):
    st.session_state["logs"].append({
        "timestamp": get_timestamp(),
        "level": level.upper(),
        "message": msg,
        **data
    })

def clear_logs():
    st.session_state["logs"] = []

# =========================
# Constants
# =========================
REQUIRED_COLUMNS = [
    "FIDO", "BARCODE", "CATEGORY_HIERARCHY", "CATEGORY_1", "CATEGORY_2", 
    "CATEGORY_3", "CATEGORY_4", "DESCRIPTION", "MANUFACTURER", "BRAND", "FIDO_TYPE"
]

OUTPUT_COLUMNS = [
    "Correct Brand?", "Correct Categories?", "Vague Description?", "Suggested Brand",
    "Suggested Category 1", "Suggested Category 2", "Suggested Category 3", 
    "Suggested New Description", "Match Strength"
]

GENERIC_BRAND_TERMS = {"generic", "unknown", "n/a", "na", "misc", "private label"}

VAGUE_TERMS = [
    "assorted", "variety pack", "misc", "pack of", "item", "product", 
    "brand", "flavor", "size", "mixed", "n/a"
]

# =========================
# Utility Functions
# =========================
def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def normalize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column headers to match required format - WITH SMART CLEANUP"""
    df = df.copy()
    
    # Debug: log original column structure
    log_event("INFO", "File column analysis", 
              original_columns=df.columns.tolist()[:15],
              total_columns=len(df.columns),
              first_row_sample=df.iloc[0].tolist()[:8] if not df.empty else [])
    
    original_col_count = len(df.columns)
    
    # Step 1: Remove completely empty columns
    columns_to_keep = []
    for col in df.columns:
        col_str = str(col)
        # Keep column if it has a real name AND contains any data
        has_real_name = not col_str.startswith('Unnamed')
        has_data = not df[col].isna().all()
        
        if has_real_name or has_data:
            columns_to_keep.append(col)
    
    # Only filter if we need to remove columns
    if len(columns_to_keep) < original_col_count:
        df = df[columns_to_keep].copy()
        removed_count = original_col_count - len(columns_to_keep)
        log_event("INFO", f"Removed {removed_count} empty unnamed columns")
    
    # Step 2: Check for and remove metadata columns
    columns_to_remove = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        
        if non_null_count == 1:
            first_val = str(df[col].iloc[0]).lower() if pd.notna(df[col].iloc[0]) else ""
            metadata_keywords = ["query", "fetch", "export", "report", "extract"]
            
            if any(keyword in first_val for keyword in metadata_keywords):
                columns_to_remove.append(col)
                log_event("INFO", f"Identified metadata column to remove: {col} = '{first_val[:50]}'")
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        log_event("INFO", f"Removed {len(columns_to_remove)} metadata columns")
    
    # Step 3: Normalize all column names to uppercase and strip whitespace
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    log_event("INFO", "After column normalization", 
              normalized_columns=df.columns.tolist()[:15],
              total_columns=len(df.columns))
    
    # Step 4: Detect and handle duplicate header rows
    if len(df) > 0:
        first_row = df.iloc[0]
        expected_headers = ["FIDO", "BARCODE", "CATEGORY_HIERARCHY", "DESCRIPTION", "BRAND"]
        first_row_values = [str(val).strip().upper() for val in first_row if pd.notna(val)]
        
        matches = sum(1 for header in expected_headers if header in first_row_values)
        if matches >= 3:
            log_event("WARNING", "Detected duplicate header row in data, removing it",
                     first_row_values=first_row_values[:10])
            df = df.iloc[1:].reset_index(drop=True)
    
    # Step 5: Add missing required columns as empty strings
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            log_event("INFO", f"Added missing column: {col}")
    
    # Step 6: Final validation
    if len(df) > 0:
        key_columns = ["FIDO", "DESCRIPTION", "BRAND"]
        has_data = False
        sample_values = {}
        
        for col in key_columns:
            if col in df.columns:
                val = df[col].iloc[0]
                sample_values[col] = str(val)[:50] if pd.notna(val) else "(empty)"
                if pd.notna(val) and str(val).strip() != "" and str(val).upper() != col:
                    has_data = True
        
        if has_data:
            log_event("INFO", "Data validation passed", sample_values=sample_values)
        else:
            log_event("ERROR", "First data row appears invalid",
                     sample_values=sample_values)
    
    return df

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add output columns without creating unnecessary copies"""
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            if col in ["Correct Brand?", "Correct Categories?", "Vague Description?"]:
                df[col] = "N"
            else:
                df[col] = ""
    return df

# =========================
# File I/O Functions - SMART HEADER DETECTION
# =========================

def detect_header_row(file_bytes: bytes, file_name: str) -> int:
    """Intelligently detect which row contains the actual headers"""
    from io import BytesIO
    
    try:
        df_peek = pd.read_excel(BytesIO(file_bytes), header=None, nrows=3)
        
        if df_peek.empty or len(df_peek) < 1:
            log_event("WARNING", "File appears empty, defaulting to row 0 as headers")
            return 0
        
        first_row = df_peek.iloc[0]
        non_empty_count = first_row.notna().sum()
        total_cells = len(first_row)
        first_cell = str(first_row.iloc[0]).lower() if pd.notna(first_row.iloc[0]) else ""
        
        metadata_keywords = ["query", "fetch", "export", "report", "extract"]
        has_metadata_keyword = any(keyword in first_cell for keyword in metadata_keywords)
        
        # SCENARIO 1: Metadata row detected
        if non_empty_count <= 2 and has_metadata_keyword:
            log_event("INFO", "Detected metadata row in Row 0", 
                     first_cell=first_cell,
                     non_empty_count=non_empty_count,
                     scenario="Snowflake export with metadata")
            return 1
        
        # SCENARIO 2: Check if Row 0 looks like actual headers
        if non_empty_count >= 5:
            header_keywords = ["fido", "barcode", "category", "description", "brand", "manufacturer"]
            row_text = " ".join([str(cell).lower() for cell in first_row if pd.notna(cell)])
            looks_like_headers = any(keyword in row_text for keyword in header_keywords)
            
            if looks_like_headers:
                log_event("INFO", "Detected headers in Row 0",
                         non_empty_count=non_empty_count,
                         scenario="Clean file with headers in Row 0")
                return 0
        
        log_event("INFO", "Using Row 0 as headers (default behavior)",
                 non_empty_count=non_empty_count)
        return 0
        
    except Exception as e:
        log_event("ERROR", f"Error detecting header row: {str(e)}, defaulting to row 0")
        return 0

@st.cache_data(show_spinner=False)
def read_excel_file(file_path_or_buffer) -> Optional[pd.DataFrame]:
    """Read Excel file for brand reference"""
    try:
        df = pd.read_excel(file_path_or_buffer)
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        string_columns = ["BRAND", "MANUFACTURER", "WEBSITE", "ALLOWED_PRODUCT_TYPES"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        log_event("INFO", "Excel file read successfully", rows=len(df), columns=len(df.columns))
        return df
    except Exception as e:
        log_event("ERROR", f"Failed to read Excel file: {str(e)}")
        return None

def read_user_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read user uploaded file with smart header detection"""
    from io import BytesIO, StringIO
    
    try:
        file_name_lower = file_name.lower()
        
        if file_name_lower.endswith('.csv'):
            try:
                content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = file_bytes.decode('latin-1')
                log_event("INFO", "Used latin-1 encoding for CSV file")
            
            df = pd.read_csv(StringIO(content))
        else:
            header_row = detect_header_row(file_bytes, file_name)
            df = pd.read_excel(BytesIO(file_bytes), header=header_row)
            log_event("INFO", f"Read Excel file with header at row {header_row}")
        
        log_event("INFO", f"Successfully loaded file: {file_name}", 
                  rows=len(df), columns=len(df.columns))
        return df
        
    except Exception as e:
        log_event("ERROR", f"Failed to read user file: {str(e)}")
        raise Exception(f"Could not read file '{file_name}': {str(e)}")

# =========================
# Brand Reference Functions
# =========================
def load_brand_reference(show_upload: bool = True) -> Optional[pd.DataFrame]:
    brand_df = read_excel_file("All_Brands_Manufacturers.xlsx")
    
    if brand_df is None and show_upload:
        st.sidebar.markdown("##### Brand-Manufacturer Reference")
        uploaded_ref = st.sidebar.file_uploader(
            "Upload All_Brands_Manufacturers.xlsx",
            type=["xlsx", "xls"],
            key="brand_reference_uploader"
        )
        
        if uploaded_ref is not None:
            brand_df = read_excel_file(uploaded_ref)
            if brand_df is None:
                st.sidebar.error("Failed to read reference file.")
    
    if brand_df is not None:
        log_event("INFO", "Brand reference loaded", brands=len(brand_df))
    
    return brand_df

@st.cache_data(show_spinner=False)
def get_manufacturer_brand_lists(brand_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    if brand_df is None or brand_df.empty:
        return [], {}
    
    if "MANUFACTURER" not in brand_df.columns or "BRAND" not in brand_df.columns:
        log_event("ERROR", "Brand reference file missing required columns")
        return [], {}
    
    manufacturer_counts = brand_df["MANUFACTURER"].value_counts()
    top_50_manufacturers = manufacturer_counts.head(50).index.tolist()
    
    brands_by_manufacturer = (
        brand_df.dropna(subset=["MANUFACTURER", "BRAND"])
        .groupby("MANUFACTURER")["BRAND"]
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )
    
    return top_50_manufacturers, brands_by_manufacturer

def get_master_brand_set(brand_df: Optional[pd.DataFrame]) -> set:
    if brand_df is None or "BRAND" not in brand_df.columns:
        return set()
    return set(brand_df["BRAND"].astype(str).str.strip().str.lower())

def get_allowed_product_types(brand_df: pd.DataFrame, brand_name: str) -> List[str]:
    """Get allowed product types for a specific brand"""
    if brand_df is None or "ALLOWED_PRODUCT_TYPES" not in brand_df.columns:
        return []
    
    brand_rows = brand_df[brand_df["BRAND"].str.lower() == brand_name.lower()]
    if brand_rows.empty:
        return []
    
    allowed_types_raw = safe_str(brand_rows.iloc[0].get("ALLOWED_PRODUCT_TYPES", ""))
    if not allowed_types_raw or allowed_types_raw == "nan":
        return []
    
    return [t.strip().lower() for t in allowed_types_raw.split(",") if t.strip()]

# =========================
# OpenAI Integration Functions
# =========================
def get_openai_client():
    """Get OpenAI client with API key from secrets"""
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            log_event("ERROR", "OpenAI API key not found in secrets")
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        log_event("ERROR", "OpenAI library not installed. Run: pip install openai")
        return None
    except Exception as e:
        log_event("ERROR", f"Failed to initialize OpenAI client: {str(e)}")
        return None

def get_brand_website(brand_df: pd.DataFrame, brand_name: str) -> str:
    """Get website URL for a specific brand"""
    if brand_df is None or "WEBSITE" not in brand_df.columns:
        return ""
    
    brand_rows = brand_df[brand_df["BRAND"].str.lower() == brand_name.lower()]
    if brand_rows.empty:
        return ""
    
    website = safe_str(brand_rows.iloc[0].get("WEBSITE", ""))
    return website if website != "nan" else ""

def check_brand_with_llm(
    description: str, 
    brand_name: str,
    manufacturer: str,
    allowed_types: List[str],
    barcode: str = "",
    website: str = ""
) -> Tuple[bool, str, float]:
    """
    Use GPT-4o to determine if a product belongs to a brand
    Returns: (belongs_to_brand, reasoning, confidence_score)
    """
    client = get_openai_client()
    if not client:
        return False, "OpenAI not available", 0.0
    
    try:
        # Build website context only if available
        website_context = f"- Website: {website}" if website else ""
        
        prompt = f"""You are a product classification expert verifying brand ownership.

Product Information:
- Description: "{description}"
- Barcode: {barcode if barcode else "Not available"}

Brand to Verify:
- Brand: {brand_name}
- Manufacturer: {manufacturer}
{website_context}
- Allowed Product Types: {', '.join(allowed_types) if allowed_types else 'Not specified'}

Analyze in this order:
1. Does the description mention "{brand_name}" or recognizable variations of the brand name?
2. If no brand mentioned and barcode is available, does the barcode help identify this as {brand_name}?
3. Does the product match one of the allowed product types?
4. Is there any competing brand name mentioned in the description?

Use MODERATE strictness:
- Brand name clearly in description → High confidence YES
- Generic description + correct product type + no competitor + helpful barcode → Acceptable YES
- Wrong product type OR competitor brand mentioned → NO

Respond in this exact format:
BELONGS: YES or NO
CONFIDENCE: 0.0 to 1.0
REASONING: Brief explanation (one sentence)"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a product taxonomy expert who helps classify products accurately. Use moderate strictness when evaluating brand ownership."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse response
        belongs = "YES" in result.split("BELONGS:")[1].split("\n")[0].upper()
        
        confidence_line = result.split("CONFIDENCE:")[1].split("\n")[0].strip()
        confidence = float(confidence_line.split()[0])
        
        reasoning = result.split("REASONING:")[1].strip()
        
        log_event("INFO", f"LLM check: {brand_name}", 
                 belongs=belongs, 
                 confidence=confidence,
                 reasoning=reasoning[:100])
        
        return belongs, reasoning, confidence
        
    except Exception as e:
        log_event("ERROR", f"LLM check failed: {str(e)}")
        return False, f"Error: {str(e)}", 0.0

def estimate_llm_cost(num_products: int, model: str = "gpt-4o") -> float:
    """Estimate cost for LLM checks"""
    # Rough estimates per product check
    costs_per_product = {
        "gpt-4o": 0.005,      # ~$0.005 per product
        "gpt-4o-mini": 0.0005  # ~$0.0005 per product
    }
    return num_products * costs_per_product.get(model, 0.005)

# =========================
# Brand Processing Functions
# =========================
def smart_pattern_match(description: str, allowed_types: List[str]) -> Tuple[bool, float, str]:
    """
    Improved pattern matching with plural/singular handling
    Returns: (match_found, confidence, matched_type)
    """
    if not allowed_types:
        return False, 0.0, ""
    
    desc_lower = description.lower()
    
    for product_type in allowed_types:
        pt_lower = product_type.lower().strip()
        
        # Direct match - highest confidence
        if pt_lower in desc_lower:
            return True, 0.95, product_type
        
        # Handle plural/singular variations
        # Remove trailing 's' and check
        if pt_lower.endswith('s'):
            singular = pt_lower[:-1]
            if singular in desc_lower:
                return True, 0.90, product_type
        else:
            plural = pt_lower + 's'
            if plural in desc_lower:
                return True, 0.90, product_type
        
        # Check if all words from product_type appear in description
        words = pt_lower.split()
        if len(words) > 1:
            all_words_present = all(word in desc_lower for word in words)
            if all_words_present:
                return True, 0.85, product_type
        
        # Check for common variations
        variations = {
            'chip': ['chips', 'chip', 'crisps'],
            'drink': ['drinks', 'drink', 'beverage', 'beverages'],
            'snack': ['snacks', 'snack'],
            'powder': ['powder', 'powders'],
            'supplement': ['supplement', 'supplements'],
            'energy': ['energy', 'energizing'],
            'protein': ['protein', 'proteins']
        }
        
        # Check if any key word has variations in description
        for base_word, var_list in variations.items():
            if base_word in pt_lower:
                for variant in var_list:
                    if variant in desc_lower:
                        return True, 0.80, product_type
    
    return False, 0.0, ""
    """Extract brand from description"""
    desc_lower = description.lower()
    
    best_brand = None
    best_length = 0
    
    for brand in master_brands:
        if len(brand) > 3 and brand in desc_lower:
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, desc_lower) and len(brand) > best_length:
                best_brand = brand
                best_length = len(brand)
    
    if best_brand:
        return best_brand.title()
    
    return None

def detect_product_family(description: str) -> Optional[str]:
    desc_lower = description.lower()
    
    if any(term in desc_lower for term in ["wine", "merlot", "cabernet", "pinot", "chardonnay"]):
        return "Wine"
    if any(term in desc_lower for term in ["coffee", "espresso", "k-cup", "keurig"]):
        return "Coffee"
    if any(term in desc_lower for term in ["vitamin", "supplement", "capsule", "tablet"]):
        return "Supplements"
    if any(term in desc_lower for term in ["decoration", "ornament", "halloween"]):
        return "Decor"
    if any(term in desc_lower for term in ["fitness", "treadmill", "exercise"]):
        return "Fitness"
    
    return None

# =========================
# Main Cleanup Functions
# =========================
def brand_accuracy_cleanup(
    df: pd.DataFrame,
    brand_df: Optional[pd.DataFrame],
    selected_manufacturer: str,
    selected_brand: str,
    log_changes_only: bool = True,
    max_logs: int = 200,
    use_llm: bool = True
) -> pd.DataFrame:
    
    if not selected_manufacturer or not selected_brand:
        log_event("ERROR", "No manufacturer or brand selected")
        return df
    
    log_event("INFO", "Starting brand accuracy cleanup", 
              manufacturer=selected_manufacturer, 
              brand=selected_brand, 
              rows=len(df),
              llm_enabled=use_llm)
    
    ensure_output_columns(df)
    
    master_brands = get_master_brand_set(brand_df)
    allowed_types = get_allowed_product_types(brand_df, selected_brand)
    brand_website = get_brand_website(brand_df, selected_brand)
    selected_brand_lower = selected_brand.lower()
    
    log_event("INFO", "Brand matching setup", 
              selected_brand=selected_brand,
              manufacturer=selected_manufacturer,
              allowed_types=allowed_types,
              website=brand_website if brand_website else "Not available",
              master_brands_count=len(master_brands))
    
    # First pass: identify uncertain cases
    uncertain_indices = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        df["Correct Brand?"] = "N"
        df["Suggested Brand"] = ""
        df["Match Strength"] = ""
        
        total_rows = len(df)
        
        # Pass 1: Pattern matching
        for idx, row in df.iterrows():
            if idx % 100 == 0 or idx == total_rows - 1:
                progress = (idx + 1) / (total_rows * 2)  # First half of progress
                progress_bar.progress(progress)
                status_text.text(f"Pattern matching: {idx + 1} of {total_rows}")
            
            description = safe_str(row.get("DESCRIPTION", ""))
            
            # Try smart pattern matching
            match_found, confidence, matched_type = smart_pattern_match(description, allowed_types)
            
            if match_found and confidence >= 0.85:
                # High confidence match - mark as correct, skip LLM
                df.at[idx, "Correct Brand?"] = "Y"
                df.at[idx, "Match Strength"] = "High"
                df.at[idx, "Suggested Brand"] = ""
            elif match_found and confidence >= 0.70 and use_llm:
                # Medium confidence - needs LLM verification
                uncertain_indices.append(idx)
                df.at[idx, "Match Strength"] = "Uncertain"
            elif not match_found and use_llm:
                # No match found - needs LLM check
                uncertain_indices.append(idx)
                df.at[idx, "Match Strength"] = "Uncertain"
            else:
                # No match and no LLM - mark as incorrect
                df.at[idx, "Correct Brand?"] = "N"
                df.at[idx, "Match Strength"] = "Low"
        
        # Show how many need AI review (no cost, just count)
        if use_llm and uncertain_indices:
            st.info(f"🤖 {len(uncertain_indices)} products need AI verification")
            
            # Pass 2: LLM verification for uncertain cases
            llm_checks = 0
            for idx in uncertain_indices:
                progress = 0.5 + ((llm_checks + 1) / len(uncertain_indices)) * 0.5
                progress_bar.progress(progress)
                status_text.text(f"AI verification: {llm_checks + 1} of {len(uncertain_indices)}")
                
                row = df.iloc[idx]
                description = safe_str(row.get("DESCRIPTION", ""))
                barcode = safe_str(row.get("BARCODE", ""))
                
                belongs, reasoning, llm_confidence = check_brand_with_llm(
                    description=description,
                    brand_name=selected_brand,
                    manufacturer=selected_manufacturer,
                    allowed_types=allowed_types,
                    barcode=barcode,
                    website=brand_website
                )
                
                if belongs:
                    df.at[idx, "Correct Brand?"] = "Y"
                    df.at[idx, "Match Strength"] = f"AI-Verified ({llm_confidence:.0%})"
                else:
                    df.at[idx, "Correct Brand?"] = "N"
                    df.at[idx, "Match Strength"] = f"AI-Rejected ({llm_confidence:.0%})"
                    
                    # Try to suggest alternative brand
                    suggested_brand = extract_brand_from_description(description, master_brands)
                    if suggested_brand and suggested_brand.lower() != selected_brand_lower:
                        df.at[idx, "Suggested Brand"] = suggested_brand
                
                llm_checks += 1
                
                # Rate limiting - avoid hitting API too fast
                if llm_checks < len(uncertain_indices):
                    time.sleep(0.2)
        
        correct_count = len(df[df["Correct Brand?"] == "Y"])
        incorrect_count = len(df[df["Correct Brand?"] == "N"])
        
        log_event("INFO", "Brand cleanup completed",
                  correct_brands=correct_count,
                  incorrect_brands=incorrect_count,
                  llm_checks=len(uncertain_indices) if use_llm else 0,
                  total_rows=len(df))
        
        return df
        
    except Exception as e:
        log_event("ERROR", f"Brand cleanup error: {str(e)}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty()

def category_hierarchy_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    log_event("INFO", "Starting category hierarchy cleanup", rows=len(df))
    
    ensure_output_columns(df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % 100 == 0 or idx == total_rows - 1:
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Analyzing categories for row {idx + 1} of {total_rows}")
            
            current_cats = (
                safe_str(row.get("CATEGORY_1", "")),
                safe_str(row.get("CATEGORY_2", "")),
                safe_str(row.get("CATEGORY_3", "")),
                safe_str(row.get("CATEGORY_4", ""))
            )
            
            all_categories_filled = all(cat for cat in current_cats)
            
            if all_categories_filled:
                df.at[idx, "Correct Categories?"] = "Y"
            else:
                df.at[idx, "Correct Categories?"] = "N"
                df.at[idx, "Match Strength"] = "Medium"
        
        correct_count = len(df[df["Correct Categories?"] == "Y"])
        log_event("INFO", "Category cleanup completed", 
                  correct_categories=correct_count,
                  total_rows=len(df))
        
        return df
        
    except Exception as e:
        log_event("ERROR", f"Category cleanup error: {str(e)}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty()

def vague_description_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    log_event("INFO", "Starting description cleanup", rows=len(df))
    
    ensure_output_columns(df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_rows = len(df)
        vague_pattern = r'\b(' + '|'.join(re.escape(term) for term in VAGUE_TERMS) + r')\b'
        
        for idx, row in df.iterrows():
            if idx % 100 == 0 or idx == total_rows - 1:
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Analyzing description for row {idx + 1} of {total_rows}")
            
            description = safe_str(row.get("DESCRIPTION", ""))
            
            is_vague = bool(re.search(vague_pattern, description.lower())) or len(description.strip()) < 10
            
            df.at[idx, "Vague Description?"] = "Y" if is_vague else "N"
            
            if is_vague:
                df.at[idx, "Match Strength"] = "Medium"
        
        vague_count = len(df[df["Vague Description?"] == "Y"])
        log_event("INFO", "Description cleanup completed",
                  vague_descriptions=vague_count,
                  total_rows=len(df))
        
        return df
        
    except Exception as e:
        log_event("ERROR", f"Description cleanup error: {str(e)}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty()

# =========================
# UI Functions
# =========================
def render_sidebar_controls(brand_df: Optional[pd.DataFrame]):
    st.sidebar.header("📁 File Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload data file from Snowflake export",
        type=["xlsx", "xls", "csv"],
        help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
    raw_df = None
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            file_key = f"{uploaded_file.name}:{len(file_bytes)}"
            
            if st.session_state.get("_last_file_key") != file_key:
                st.session_state["_last_file_key"] = file_key
                st.session_state["_raw_df"] = read_user_file(uploaded_file.name, file_bytes)
                st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
            
            raw_df = st.session_state.get("_raw_df")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            log_event("ERROR", f"File reading error: {str(e)}")
    
    st.sidebar.header("🔧 Cleanup Options")
    
    cleanup_type = st.sidebar.selectbox(
        "Select cleanup type:",
        ["Brand Accuracy", "Category Hierarchy Cleanup", "Vague Description Cleanup"],
        help="Choose the type of data cleanup to perform"
    )
    
    selected_manufacturer = None
    selected_brand = None
    
    if cleanup_type == "Brand Accuracy":
        st.sidebar.subheader("Brand Settings")
        
        if brand_df is not None and not brand_df.empty:
            show_top_only = st.sidebar.checkbox("Show only top 50 manufacturers", value=True)
            
            top_manufacturers, brands_by_manufacturer = get_manufacturer_brand_lists(brand_df)
            
            if not top_manufacturers and not brands_by_manufacturer:
                st.sidebar.error("❌ No valid manufacturer data found in reference file.")
            else:
                manufacturer_options = top_manufacturers if show_top_only else sorted(brands_by_manufacturer.keys())
                
                if manufacturer_options:
                    selected_manufacturer = st.sidebar.selectbox(
                        "Select manufacturer:",
                        options=manufacturer_options,
                        help="Choose the manufacturer to focus cleanup on"
                    )
                    
                    if selected_manufacturer and selected_manufacturer in brands_by_manufacturer:
                        brand_options = brands_by_manufacturer[selected_manufacturer]
                        if brand_options:
                            selected_brand = st.sidebar.selectbox(
                                "Select brand:",
                                options=brand_options,
                                help="Choose the specific brand for cleanup"
                            )
                        else:
                            st.sidebar.warning("⚠️ No brands found for selected manufacturer.")
                else:
                    st.sidebar.error("❌ No manufacturers available.")
        else:
            st.sidebar.warning("⚠️ No brand reference data available.")
    
    st.sidebar.header("⚙️ Advanced Settings")
    
    # Add LLM toggle
    use_llm = st.sidebar.checkbox(
        "🤖 Use AI for uncertain cases",
        value=True,
        help="Use GPT-4o to verify products when pattern matching is uncertain (costs ~$0.005 per product)"
    )
    
    log_changes_only = st.sidebar.checkbox(
        "Log only changes",
        value=True,
        help="Only log rows that were flagged or changed"
    )
    
    max_logs = st.sidebar.number_input(
        "Maximum log entries:",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )
    
    run_cleanup = st.sidebar.button(
        "🚀 Run Cleanup",
        use_container_width=True,
        type="primary"
    )
    
    clear_form = st.sidebar.button(
        "🗑️ Clear Form", 
        use_container_width=True,
        help="Clear uploaded file and reset form"
    )
    
    if clear_form:
        keys_to_clear = ["_last_file_key", "_raw_df"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    return (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
            selected_brand, log_changes_only, max_logs, run_cleanup, use_llm)

def render_results(df: pd.DataFrame):
    st.subheader("📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        if "Correct Brand?" in df.columns:
            correct_brands = len(df[df["Correct Brand?"] == "Y"])
            st.metric("Correct Brands", f"{correct_brands:,}")
    
    with col3:
        if "Correct Categories?" in df.columns:
            correct_categories = len(df[df["Correct Categories?"] == "Y"])
            st.metric("Correct Categories", f"{correct_categories:,}")
    
    with col4:
        if "Suggested Brand" in df.columns:
            suggestions = len(df[df["Suggested Brand"] != ""])
            st.metric("Suggestions Made", f"{suggestions:,}")
    
    st.dataframe(df.head(100), use_container_width=True)
    
    if len(df) > 100:
        st.info(f"Showing first 100 rows of {len(df):,} total rows. Download the full file to see all data.")
    
    try:
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        
        st.download_button(
            "📥 Download Cleaned File (.xlsx)",
            data=output_buffer.getvalue(),
            file_name=f"taxonomy_guardian_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Error preparing download: {str(e)}")
        log_event("ERROR", f"Download preparation error: {str(e)}")

def render_logs():
    with st.expander("📋 Application Logs", expanded=False):
        logs = st.session_state.get("logs", [])
        
        if logs:
            recent_logs = logs[-50:]
            
            for log_entry in reversed(recent_logs):
                timestamp = log_entry.get("timestamp", "")
                level = log_entry.get("level", "INFO")
                message = log_entry.get("message", "")
                
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%H:%M:%S")
                except:
                    formatted_time = timestamp[:8] if len(timestamp) > 8 else timestamp
                
                if level == "ERROR":
                    st.error(f"[{formatted_time}] {message}")
                elif level == "WARNING":
                    st.warning(f"[{formatted_time}] {message}")
                elif level == "SUCCESS":
                    st.success(f"[{formatted_time}] {message}")
                else:
                    st.info(f"[{formatted_time}] {message}")
        else:
            st.info("No logs available")
        
        if st.button("🗑️ Clear Logs"):
            clear_logs()
            st.rerun()

# =========================
# Main Application
# =========================
def main():
    try:
        brand_df = load_brand_reference(show_upload=True)
        
        (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
         selected_brand, log_changes_only, max_logs, run_cleanup, use_llm) = render_sidebar_controls(brand_df)
        
        st.header("🛡️ Data Cleanup Status")
        
        if uploaded_file is None:
            st.info("👆 Upload a data file in the sidebar to get started")
            
            with st.expander("ℹ️ How to Use Taxonomy Guardian", expanded=True):
                st.markdown("""
                **Step 1:** Upload your data file (Excel or CSV) from Snowflake export
                
                **Step 2:** Select the type of cleanup you need:
                - **Brand Accuracy**: Validates and suggests brand corrections
                - **Category Hierarchy**: Ensures all 4 category levels are properly filled
                - **Vague Description**: Identifies and improves unclear product descriptions
                
                **Step 3:** For Brand Accuracy, select your target manufacturer and brand
                
                **Step 4:** Click "Run Cleanup" to process your data
                
                **Step 5:** Review results and download the cleaned file
                """)
            
            render_logs()
            return
        
        if raw_df is None:
            st.error("❌ Failed to load the uploaded file.")
            render_logs()
            return
        
        st.success(f"✅ File loaded: **{uploaded_file.name}**")
        
        file_col1, file_col2 = st.columns(2)
        with file_col1:
            st.metric("Rows", f"{len(raw_df):,}")
        with file_col2:
            st.metric("Columns", len(raw_df.columns))
        
        try:
            processed_df = normalize_column_headers(raw_df)
            
            if processed_df.empty:
                st.error("❌ File appears to be empty after processing.")
                render_logs()
                return
            
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_df.columns]
            
            if missing_columns:
                st.warning(f"⚠️ Missing required columns (added as empty): {', '.join(missing_columns)}")
            else:
                st.success("✅ All required columns are present")
            
            key_columns = ["FIDO", "DESCRIPTION", "BRAND"]
            data_check = {}
            for col in key_columns:
                if col in processed_df.columns:
                    non_empty = processed_df[col].notna().sum()
                    data_check[col] = non_empty
            
            total_rows = len(processed_df)
            if data_check:
                col1_check, col2_check, col3_check = st.columns(3)
                with col1_check:
                    fido_count = data_check.get("FIDO", 0)
                    st.metric("FIDO values", f"{fido_count}/{total_rows}")
                with col2_check:
                    desc_count = data_check.get("DESCRIPTION", 0)
                    st.metric("Description values", f"{desc_count}/{total_rows}")
                with col3_check:
                    brand_count = data_check.get("BRAND", 0)
                    st.metric("Brand values", f"{brand_count}/{total_rows}")
                
                if any(count < total_rows * 0.5 for count in data_check.values()):
                    st.warning("⚠️ Some columns have significant missing data. Please check the logs for details.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            log_event("ERROR", f"File processing error: {str(e)}")
            render_logs()
            return
        
        st.subheader(f"🔧 Selected Cleanup: {cleanup_type}")
        
        if cleanup_type == "Brand Accuracy":
            if not selected_manufacturer or not selected_brand:
                st.warning("⚠️ Please select both manufacturer and brand in the sidebar")
                render_logs()
                return
            else:
                st.success(f"🎯 Target: **{selected_manufacturer}** → **{selected_brand}**")
        
        if not run_cleanup:
            st.info("👆 Click **Run Cleanup** in the sidebar to process your data")
            render_logs()
            return
        
        try:
            log_event("INFO", f"Starting {cleanup_type} cleanup", 
                      rows=len(processed_df),
                      cleanup_type=cleanup_type)
            
            with st.spinner(f"🔄 Running {cleanup_type.lower()}..."):
                
                if cleanup_type == "Brand Accuracy":
                    cleaned_df = brand_accuracy_cleanup(
                        processed_df, 
                        brand_df, 
                        selected_manufacturer, 
                        selected_brand,
                        log_changes_only, 
                        max_logs,
                        use_llm
                    )
                
                elif cleanup_type == "Category Hierarchy Cleanup":
                    cleaned_df = category_hierarchy_cleanup(processed_df)
                
                elif cleanup_type == "Vague Description Cleanup":
                    cleaned_df = vague_description_cleanup(processed_df)
                
                else:
                    st.error("❌ Unknown cleanup type selected")
                    return
            
            st.success("✅ Cleanup completed successfully!")
            log_event("SUCCESS", f"{cleanup_type} cleanup completed successfully")
            
            render_results(cleaned_df)
        
        except Exception as e:
            error_msg = f"Error during {cleanup_type.lower()}: {str(e)}"
            st.error(f"❌ {error_msg}")
            log_event("ERROR", error_msg)
        
        render_logs()
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        log_event("ERROR", f"Application error: {str(e)}")
        render_logs()

if __name__ == "__main__":
    main()
