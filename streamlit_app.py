# streamlit_app.py
# Taxonomy Guardian v27 - With seltzer debug logging

import io
import re
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# Streamlit import
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

# App config
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

# Logging System
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

# Constants
REQUIRED_COLUMNS = [
    "FIDO", "BARCODE", "CATEGORY_HIERARCHY", "CATEGORY_1", "CATEGORY_2", 
    "CATEGORY_3", "CATEGORY_4", "DESCRIPTION", "MANUFACTURER", "BRAND", "FIDO_TYPE"
]

OUTPUT_COLUMNS = [
    "Correct Brand?", "Correct Categories?", "Vague Description?", "Suggested Brand",
    "Suggested Category 1", "Suggested Category 2", "Suggested Category 3", 
    "Suggested New Description", "Match Strength"
]

VAGUE_TERMS = [
    "assorted", "variety pack", "misc", "pack of", "item", "product", 
    "brand", "flavor", "size", "mixed", "n/a"
]

# Utility Functions
def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            if col in ["Correct Brand?", "Correct Categories?", "Vague Description?"]:
                df[col] = "N"
            else:
                df[col] = ""
    return df

# OpenAI Integration - Two-Stage LLM Approach

def get_openai_client():
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            log_event("ERROR", "OpenAI API key not found in secrets")
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        log_event("ERROR", "OpenAI library not installed")
        return None
    except Exception as e:
        log_event("ERROR", f"Failed to initialize OpenAI: {str(e)}")
        return None

def get_master_brands_sample(master_brands: set, limit: int = 50) -> List[str]:
    """Get a sample of brands from the master list for LLM context"""
    if not master_brands:
        return []
    # Return top N brands alphabetically for consistency
    return sorted(list(master_brands))[:limit]

def verify_brand_with_llm(
    description: str,
    brand_name: str,
    manufacturer: str,
    allowed_types: List[str],
    barcode: str = "",
    website: str = ""
) -> Tuple[bool, str, float]:
    """Stage 1: Verify if product belongs to specified brand"""
    client = get_openai_client()
    if not client:
        return False, "OpenAI not available", 0.0
    
    try:
        website_info = f"Website: {website}" if website else "Website: N/A"
        
        prompt = f"""You are a product taxonomy expert verifying brand ownership.

PRODUCT:
Description: "{description}"
Barcode: {barcode if barcode else "N/A"}

BRAND TO VERIFY:
Brand: {brand_name}
Manufacturer: {manufacturer}
{website_info}
Allowed Products: {', '.join(allowed_types) if allowed_types else "Not specified"}

VERIFICATION PRIORITY:
1. Check if description clearly mentions "{brand_name}" or recognizable variations
2. If description is vague/unclear, consider if barcode suggests this manufacturer
3. Verify the product type matches one of the allowed products
4. Check for competing brand names in the description

GUIDELINES:
- Description clearly mentions {brand_name} → YES
- Vague description + barcode suggests right manufacturer + correct product type → LIKELY YES
- Different brand name mentioned → NO
- Wrong product type → NO
- Generic unbranded item (BANANAS, MILK, EGGS) → NO

RESPOND:
BELONGS: YES or NO
CONFIDENCE: 0.0 to 1.0
REASONING: One sentence (mention if you used barcode evidence)"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a product taxonomy expert. Analyze carefully and use barcode context when descriptions are unclear."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        belongs = "YES" in result.split("BELONGS:")[1].split("\n")[0].upper()
        confidence_line = result.split("CONFIDENCE:")[1].split("\n")[0].strip()
        confidence = float(confidence_line.split()[0])
        reasoning = result.split("REASONING:")[1].strip()
        
        return belongs, reasoning, confidence
        
    except Exception as e:
        log_event("ERROR", f"Stage 1 LLM verification failed: {str(e)}")
        return False, f"Error: {str(e)}", 0.0

def suggest_brand_with_llm(
    description: str,
    barcode: str,
    categories: List[str],
    rejected_from_brand: str,
    master_brands_sample: List[str]
) -> Tuple[str, str, float]:
    """Stage 2: Suggest correct brand if verification failed"""
    client = get_openai_client()
    if not client:
        return "Unknown", "OpenAI not available", 0.0
    
    try:
        category_hierarchy = " > ".join([c for c in categories[:3] if c])
        brands_list = ", ".join(master_brands_sample[:50]) if master_brands_sample else "Not available"
        
        prompt = f"""You are a brand identification expert.

PRODUCT:
Description: "{description}"
Barcode: {barcode if barcode else "N/A"}
Category: {category_hierarchy if category_hierarchy else "Unknown"}
Rejected from: {rejected_from_brand}

KNOWN BRANDS IN SYSTEM:
{brands_list}

IDENTIFICATION PRIORITY:
1. **PRIMARY**: Look for explicit brand names in the description
   - Capitalized words at the start often indicate brands
   - Match against the known brands list when possible
2. **SECONDARY**: If description is vague (e.g., "SELTZER 12PK"), use barcode context
   - First 6-10 digits identify the manufacturer
   - Use your knowledge of barcode patterns for major brands
3. **TERTIARY**: Use category context to validate your suggestion
   - Does the suggested brand make sense for this product category?

RULES:
- Extract ONLY the brand name (no sizes, quantities, or descriptors)
  Good: "Coca-Cola" | Bad: "Coca-Cola 12oz Can"
- If it's a generic unbranded item (BANANAS, EGGS, MILK) → return "GENERIC_ITEM"
- If you see a brand not in the known brands list → return it anyway
- If completely uncertain → return "Unknown"

RESPOND:
SUGGESTED_BRAND: [brand name, "GENERIC_ITEM", or "Unknown"]
CONFIDENCE: 0.0 to 1.0
REASONING: One sentence (mention if you used barcode or category context)"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a brand identification expert. Use all available context - description, barcode patterns, and category information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        suggested_brand = result.split("SUGGESTED_BRAND:")[1].split("\n")[0].strip()
        confidence_line = result.split("CONFIDENCE:")[1].split("\n")[0].strip()
        confidence = float(confidence_line.split()[0])
        reasoning = result.split("REASONING:")[1].strip()
        
        return suggested_brand, reasoning, confidence
        
    except Exception as e:
        log_event("ERROR", f"Stage 2 LLM suggestion failed: {str(e)}")
        return "Unknown", f"Error: {str(e)}", 0.0

# File I/O
def detect_header_row(file_bytes: bytes, file_name: str) -> int:
    from io import BytesIO
    
    try:
        df_peek = pd.read_excel(BytesIO(file_bytes), header=None, nrows=3)
        
        if df_peek.empty or len(df_peek) < 1:
            return 0
        
        first_row = df_peek.iloc[0]
        non_empty_count = first_row.notna().sum()
        first_cell = str(first_row.iloc[0]).lower() if pd.notna(first_row.iloc[0]) else ""
        
        metadata_keywords = ["query", "fetch", "export", "report", "extract"]
        has_metadata_keyword = any(keyword in first_cell for keyword in metadata_keywords)
        
        if non_empty_count <= 2 and has_metadata_keyword:
            log_event("INFO", "Detected metadata row in Row 0")
            return 1
        
        if non_empty_count >= 5:
            header_keywords = ["fido", "barcode", "category", "description", "brand", "manufacturer"]
            row_text = " ".join([str(cell).lower() for cell in first_row if pd.notna(cell)])
            looks_like_headers = any(keyword in row_text for keyword in header_keywords)
            
            if looks_like_headers:
                log_event("INFO", "Detected headers in Row 0")
                return 0
        
        return 0
        
    except Exception as e:
        log_event("ERROR", f"Error detecting header row: {str(e)}")
        return 0

def read_excel_file(file_path_or_buffer) -> Optional[pd.DataFrame]:
    """Read Excel file for brand reference - NO CACHING to always get latest data"""
    try:
        df = pd.read_excel(file_path_or_buffer)
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        string_columns = ["BRAND", "MANUFACTURER", "WEBSITE", "ALLOWED_PRODUCT_TYPES"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        log_event("ERROR", f"Failed to read Excel file: {str(e)}")
        return None

def read_user_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO, StringIO
    
    try:
        file_name_lower = file_name.lower()
        
        if file_name_lower.endswith('.csv'):
            try:
                content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = file_bytes.decode('latin-1')
            df = pd.read_csv(StringIO(content))
        else:
            header_row = detect_header_row(file_bytes, file_name)
            df = pd.read_excel(BytesIO(file_bytes), header=header_row)
        
        log_event("INFO", f"Successfully loaded file: {file_name}", rows=len(df), columns=len(df.columns))
        return df
        
    except Exception as e:
        log_event("ERROR", f"Failed to read user file: {str(e)}")
        raise Exception(f"Could not read file '{file_name}': {str(e)}")

def normalize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    log_event("INFO", "File column analysis", 
              original_columns=df.columns.tolist()[:15],
              total_columns=len(df.columns))
    
    original_col_count = len(df.columns)
    
    # Remove empty unnamed columns
    columns_to_keep = []
    for col in df.columns:
        col_str = str(col)
        has_real_name = not col_str.startswith('Unnamed')
        has_data = not df[col].isna().all()
        
        if has_real_name or has_data:
            columns_to_keep.append(col)
    
    if len(columns_to_keep) < original_col_count:
        df = df[columns_to_keep].copy()
        log_event("INFO", f"Removed {original_col_count - len(columns_to_keep)} empty columns")
    
    # Remove metadata columns
    columns_to_remove = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        
        if non_null_count == 1:
            first_val = str(df[col].iloc[0]).lower() if pd.notna(df[col].iloc[0]) else ""
            metadata_keywords = ["query", "fetch", "export", "report", "extract"]
            
            if any(keyword in first_val for keyword in metadata_keywords):
                columns_to_remove.append(col)
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        log_event("INFO", f"Removed {len(columns_to_remove)} metadata columns")
    
    # Normalize column names
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    # Detect duplicate header rows
    if len(df) > 0:
        first_row = df.iloc[0]
        expected_headers = ["FIDO", "BARCODE", "CATEGORY_HIERARCHY", "DESCRIPTION", "BRAND"]
        first_row_values = [str(val).strip().upper() for val in first_row if pd.notna(val)]
        
        matches = sum(1 for header in expected_headers if header in first_row_values)
        if matches >= 3:
            log_event("WARNING", "Detected duplicate header row, removing it")
            df = df.iloc[1:].reset_index(drop=True)
    
    # Add missing required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    return df

# Brand Reference Functions
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
    
    if brand_df is not None:
        log_event("INFO", "Brand reference loaded", brands=len(brand_df))
    
    return brand_df

@st.cache_data(show_spinner=False)
def get_manufacturer_brand_lists(brand_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    if brand_df is None or brand_df.empty:
        return [], {}
    
    if "MANUFACTURER" not in brand_df.columns or "BRAND" not in brand_df.columns:
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
    log_event("INFO", f"get_allowed_product_types called for: {brand_name}")
    
    if brand_df is None:
        log_event("ERROR", "brand_df is None")
        return []
    
    if "ALLOWED_PRODUCT_TYPES" not in brand_df.columns:
        log_event("ERROR", "ALLOWED_PRODUCT_TYPES column not found in brand reference file")
        return []
    
    brand_rows = brand_df[brand_df["BRAND"].str.lower() == brand_name.lower()]
    
    if brand_rows.empty:
        log_event("ERROR", f"No rows found for brand: {brand_name}")
        return []
    
    log_event("INFO", f"Found {len(brand_rows)} row(s) for {brand_name}")
    
    allowed_types_raw = safe_str(brand_rows.iloc[0].get("ALLOWED_PRODUCT_TYPES", ""))
    log_event("INFO", f"Raw allowed_types value: '{allowed_types_raw}'")
    
    if not allowed_types_raw or allowed_types_raw == "nan":
        log_event("WARNING", f"Allowed types empty or nan for {brand_name}")
        return []
    
    types = []
    for t in allowed_types_raw.split(","):
        cleaned = t.strip().lower()
        if cleaned and cleaned != "nan":
            types.append(cleaned)
    
    log_event("INFO", f"Parsed allowed types for {brand_name}", types=types)
    return types

def get_brand_website(brand_df: pd.DataFrame, brand_name: str) -> str:
    if brand_df is None or "WEBSITE" not in brand_df.columns:
        return ""
    
    brand_rows = brand_df[brand_df["BRAND"].str.lower() == brand_name.lower()]
    if brand_rows.empty:
        return ""
    
    website = safe_str(brand_rows.iloc[0].get("WEBSITE", ""))
    return website if website != "nan" else ""

# Brand Processing Functions
def extract_brand_from_description(description: str, master_brands: set) -> Optional[str]:
    """Extract brand from description - tries master brands first, then any capitalized words"""
    desc_lower = description.lower()
    
    # First, try to find known brands from master list
    best_brand = None
    best_length = 0
    
    for brand in master_brands:
        if len(brand) > 3 and brand in desc_lower:
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, desc_lower) and len(brand) > best_length:
                best_brand = brand
                best_length = len(brand)
    
    if best_brand:
        log_event("INFO", "✅ Brand extracted from master list", 
                 description_snippet=description[:50],
                 extracted_brand=best_brand.title())
        return best_brand.title()
    
    # If no known brand found, look for capitalized words at start (likely brand names)
    words = description.split()
    brand_words = []
    
    for word in words[:5]:  # Check first 5 words only
        skip_words = {'the', 'a', 'an', 'with', 'for', 'and', 'or', 'in', 'on', 'at', 'to', 'of'}
        if word.lower() in skip_words:
            continue
        
        if word[0].isupper() or word.isupper():
            descriptive_words = {'new', 'pack', 'box', 'case', 'original', 'organic', 'natural', 'fresh'}
            if word.lower() not in descriptive_words:
                brand_words.append(word)
                if len(brand_words) >= 2:
                    break
        else:
            if brand_words:
                break
    
    if brand_words:
        suggested = ' '.join(brand_words)
        log_event("INFO", "💡 Brand extracted from capitalization", 
                 description_snippet=description[:50],
                 extracted_brand=suggested)
        return suggested
    
    log_event("WARNING", "❌ No brand could be extracted", 
             description_snippet=description[:50])
    return None

def smart_pattern_match(description: str, allowed_types: List[str], categories: List[str] = None) -> Tuple[bool, float, str]:
    """
    Improved pattern matching with category validation
    Returns: (match_found, confidence, matched_type)
    """
    if not allowed_types:
        log_event("WARNING", "smart_pattern_match called with empty allowed_types")
        return False, 0.0, ""
    
    desc_lower = description.lower()
    
    # DEBUG: Log seltzer product details
    if "seltzer" in desc_lower:
        log_event("INFO", "🍹 SELTZER DEBUG", 
                 description=desc_lower[:80],
                 allowed_types_sample=allowed_types[:3],
                 total_types=len(allowed_types))
    
    # Extract category keywords if provided
    category_text = ""
    if categories:
        category_text = " ".join([c.lower() for c in categories if c]).strip()
    
    for product_type in allowed_types:
        pt_lower = product_type.lower().strip()
        
        # DEBUG: Log each seltzer comparison
        if "seltzer" in desc_lower and "seltzer" in pt_lower:
            contains_result = pt_lower in desc_lower
            log_event("INFO", "🔍 SELTZER COMPARISON",
                     pt_lower=f"'{pt_lower}'",
                     desc_snippet=f"'{desc_lower[:60]}'",
                     match_found=contains_result)
        
        # Direct match in description
        if pt_lower in desc_lower:
            # If we have category info, check if it BOOSTS confidence (but don't reject if no match)
            if category_text:
                type_words = pt_lower.split()
                # Check for category match to boost confidence
                category_match = any(word in category_text for word in type_words if len(word) > 3)
                if category_match:
                    return True, 0.95, product_type  # Perfect match - description + category
                else:
                    return True, 0.82, product_type  # Uncertain - send to LLM for verification
            return True, 0.90, product_type
        
        # Handle plural/singular variations
        if pt_lower.endswith('s'):
            singular = pt_lower[:-1]
            if singular in desc_lower:
                return True, 0.85, product_type
        else:
            plural = pt_lower + 's'
            if plural in desc_lower:
                return True, 0.85, product_type
        
        # Check if all words from product_type appear
        words = pt_lower.split()
        if len(words) > 1:
            all_words_present = all(word in desc_lower for word in words)
            if all_words_present:
                return True, 0.83, product_type
        
        # Check for common variations
        variations = {
            'chip': ['chips', 'chip', 'crisps', 'crisp'],
            'drink': ['drinks', 'drink', 'beverage', 'beverages'],
            'snack': ['snacks', 'snack'],
            'powder': ['powder', 'powders'],
            'supplement': ['supplement', 'supplements'],
            'energy': ['energy', 'energizing'],
            'protein': ['protein', 'proteins'],
            'wine': ['wine', 'wines', 'vino'],
            'beer': ['beer', 'beers', 'ale', 'lager'],
            'seltzer': ['seltzer', 'seltzers', 'spiked'],
            'coffee': ['coffee', 'espresso', 'cappuccino']
        }
        
        for base_word, var_list in variations.items():
            if base_word in pt_lower:
                for variant in var_list:
                    if variant in desc_lower:
                        return True, 0.80, product_type
    
    return False, 0.0, ""

# Main Cleanup Functions
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
              manufacturer=selected_manufacturer, brand=selected_brand, rows=len(df), llm_enabled=use_llm)
    
    ensure_output_columns(df)
    
    master_brands = get_master_brand_set(brand_df)
    allowed_types = get_allowed_product_types(brand_df, selected_brand)
    brand_website = get_brand_website(brand_df, selected_brand)
    selected_brand_lower = selected_brand.lower()
    
    log_event("INFO", "Brand matching setup", 
              selected_brand=selected_brand,
              manufacturer=selected_manufacturer,
              allowed_types=allowed_types,
              website=brand_website if brand_website else "Not available")
    
    uncertain_indices = []
    stage2_calls = 0
    
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
                progress = (idx + 1) / (total_rows * 2)
                progress_bar.progress(progress)
                status_text.text(f"Pattern matching: {idx + 1} of {total_rows}")
            
            description = safe_str(row.get("DESCRIPTION", ""))
            
            # Get category information for validation
            categories = [
                safe_str(row.get("CATEGORY_1", "")),
                safe_str(row.get("CATEGORY_2", "")),
                safe_str(row.get("CATEGORY_3", ""))
            ]
            
            # Try smart pattern matching with category validation
            match_found, confidence, matched_type = smart_pattern_match(description, allowed_types, categories)
            
            if match_found and confidence >= 0.85:
                df.at[idx, "Correct Brand?"] = "Y"
                df.at[idx, "Match Strength"] = "High"
                df.at[idx, "Suggested Brand"] = ""
            elif match_found and confidence >= 0.70 and use_llm:
                uncertain_indices.append(idx)
                df.at[idx, "Match Strength"] = "Uncertain"
            elif not match_found and use_llm:
                uncertain_indices.append(idx)
                df.at[idx, "Match Strength"] = "Uncertain"
            else:
                # Mark as incorrect and ALWAYS suggest alternative brand
                df.at[idx, "Correct Brand?"] = "N"
                df.at[idx, "Match Strength"] = "Low"
                
                log_event("INFO", "🔍 No pattern match - extracting suggested brand",
                         description=description[:60],
                         selected_brand=selected_brand)
                
                # Extract brand from description
                suggested_brand = extract_brand_from_description(description, master_brands)
                
                # Check if suggested brand is meaningfully different from selected brand
                if suggested_brand:
                    suggested_lower = suggested_brand.lower()
                    # Only reject if it's the exact same brand (not just contained)
                    if suggested_lower == selected_brand_lower:
                        log_event("INFO", "⚠️ Suggested brand same as selected, skipping",
                                 suggested=suggested_brand)
                        suggested_brand = None
                    elif len(suggested_brand.split()) == 1 and suggested_lower == selected_brand_lower:
                        log_event("INFO", "⚠️ Single word match with selected, skipping",
                                 suggested=suggested_brand)
                        suggested_brand = None
                
                if suggested_brand:
                    df.at[idx, "Suggested Brand"] = suggested_brand
                    log_event("INFO", "✅ Brand suggestion assigned",
                             description=description[:60],
                             suggested_brand=suggested_brand)
                else:
                    # Last resort: extract first capitalized words from description
                    desc_words = []
                    for word in description.split()[:5]:
                        if word and word[0].isupper():
                            desc_words.append(word)
                            if len(desc_words) >= 2:
                                break
                    
                    if desc_words:
                        fallback_brand = ' '.join(desc_words)
                        df.at[idx, "Suggested Brand"] = fallback_brand
                        log_event("INFO", "⚡ Fallback brand from caps",
                                 description=description[:60],
                                 fallback_brand=fallback_brand)
                    else:
                        df.at[idx, "Suggested Brand"] = "Unknown Brand"
                        log_event("WARNING", "❌ Could not extract any brand",
                                 description=description[:60])
        
        # Pass 2: LLM verification (Two-Stage Approach)
        if use_llm and uncertain_indices:
            st.info(f"🤖 {len(uncertain_indices)} products need AI verification")
            
            # Get master brands sample for Stage 2
            master_brands_sample = get_master_brands_sample(master_brands, limit=50)
            
            llm_checks = 0
            stage2_calls = 0
            
            for idx in uncertain_indices:
                progress = 0.5 + ((llm_checks + 1) / len(uncertain_indices)) * 0.5
                progress_bar.progress(progress)
                status_text.text(f"AI verification: {llm_checks + 1} of {len(uncertain_indices)}")
                
                row = df.iloc[idx]
                description = safe_str(row.get("DESCRIPTION", ""))
                barcode = safe_str(row.get("BARCODE", ""))
                
                # Get category information
                categories = [
                    safe_str(row.get("CATEGORY_1", "")),
                    safe_str(row.get("CATEGORY_2", "")),
                    safe_str(row.get("CATEGORY_3", ""))
                ]
                
                # STAGE 1: Verify if product belongs to selected brand
                log_event("INFO", "🤖 Stage 1: Verifying brand",
                         description=description[:60],
                         selected_brand=selected_brand)
                
                belongs, verify_reasoning, verify_confidence = verify_brand_with_llm(
                    description=description,
                    brand_name=selected_brand,
                    manufacturer=selected_manufacturer,
                    allowed_types=allowed_types,
                    barcode=barcode,
                    website=brand_website
                )
                
                if belongs:
                    # Stage 1 confirmed - product belongs to brand
                    df.at[idx, "Correct Brand?"] = "Y"
                    df.at[idx, "Match Strength"] = f"AI-Verified ({verify_confidence:.0%})"
                    log_event("INFO", "✅ Stage 1 complete: Brand verified",
                             description=description[:60],
                             confidence=verify_confidence,
                             reasoning=verify_reasoning[:100])
                else:
                    # Stage 1 rejected - proceed to Stage 2 for suggestion
                    df.at[idx, "Correct Brand?"] = "N"
                    df.at[idx, "Match Strength"] = f"AI-Rejected ({verify_confidence:.0%})"
                    log_event("INFO", "❌ Stage 1: Brand rejected",
                             description=description[:60],
                             reasoning=verify_reasoning[:100])
                    
                    # STAGE 2: Suggest correct brand
                    log_event("INFO", "🤖 Stage 2: Identifying correct brand",
                             description=description[:60])
                    
                    suggested_brand, suggest_reasoning, suggest_confidence = suggest_brand_with_llm(
                        description=description,
                        barcode=barcode,
                        categories=categories,
                        rejected_from_brand=selected_brand,
                        master_brands_sample=master_brands_sample
                    )
                    
                    # Only suggest if it's different from the selected brand
                    if suggested_brand and suggested_brand.lower() not in [selected_brand_lower, "unknown"]:
                        df.at[idx, "Suggested Brand"] = suggested_brand
                        log_event("INFO", "✅ Stage 2 complete: Brand suggested",
                                 suggested_brand=suggested_brand,
                                 confidence=suggest_confidence,
                                 reasoning=suggest_reasoning[:100])
                    else:
                        log_event("INFO", "⚠️ Stage 2: No alternative brand identified",
                                 result=suggested_brand)
                    
                    stage2_calls += 1
                
                llm_checks += 1
                
                # Rate limiting
                if llm_checks < len(uncertain_indices):
                    time.sleep(0.2)
        
        correct_count = len(df[df["Correct Brand?"] == "Y"])
        incorrect_count = len(df[df["Correct Brand?"] == "N"])
        
        log_event("INFO", "Brand cleanup completed",
                  correct_brands=correct_count,
                  incorrect_brands=incorrect_count,
                  stage1_verifications=len(uncertain_indices) if use_llm else 0,
                  stage2_suggestions=stage2_calls if use_llm else 0)
        
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
        log_event("INFO", "Category cleanup completed", correct_categories=correct_count)
        
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
        log_event("INFO", "Description cleanup completed", vague_descriptions=vague_count)
        
        return df
        
    except Exception as e:
        log_event("ERROR", f"Description cleanup error: {str(e)}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty()

# UI Functions
def render_sidebar_controls(brand_df: Optional[pd.DataFrame]):
    st.sidebar.header("📁 File Upload")
    
    # Initialize reset counter if not exists
    if "_reset_counter" not in st.session_state:
        st.session_state["_reset_counter"] = 0
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload data file from Snowflake export",
        type=["xlsx", "xls", "csv"],
        key=f"file_uploader_{st.session_state['_reset_counter']}"
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
    
    st.sidebar.header("🔧 Cleanup Options")
    
    cleanup_type = st.sidebar.selectbox(
        "Select cleanup type:",
        ["Brand Accuracy", "Category Hierarchy Cleanup", "Vague Description Cleanup"]
    )
    
    selected_manufacturer = None
    selected_brand = None
    
    if cleanup_type == "Brand Accuracy":
        st.sidebar.subheader("Brand Settings")
        
        if brand_df is not None and not brand_df.empty:
            top_manufacturers, brands_by_manufacturer = get_manufacturer_brand_lists(brand_df)
            
            if brands_by_manufacturer:
                # Show ALL manufacturers (not just top 50)
                manufacturer_options = sorted(brands_by_manufacturer.keys())
                
                if manufacturer_options:
                    selected_manufacturer = st.sidebar.selectbox(
                        "Select manufacturer:",
                        options=manufacturer_options
                    )
                    
                    if selected_manufacturer and selected_manufacturer in brands_by_manufacturer:
                        brand_options = brands_by_manufacturer[selected_manufacturer]
                        if brand_options:
                            selected_brand = st.sidebar.selectbox(
                                "Select brand:",
                                options=brand_options
                            )
        else:
            st.sidebar.warning("⚠️ No brand reference data available.")
    
    st.sidebar.header("⚙️ Advanced Settings")
    
    use_llm = st.sidebar.checkbox(
        "🤖 Use AI for uncertain cases",
        value=True,
        help="Use GPT-4o to verify products when pattern matching is uncertain"
    )
    
    log_changes_only = st.sidebar.checkbox(
        "Log only changes",
        value=True
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
        use_container_width=True
    )
    
    if clear_form:
        # Set flag to clear everything on next render
        st.session_state["_clear_triggered"] = True
        clear_logs()
        st.rerun()
    
    return (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
            selected_brand, log_changes_only, max_logs, run_cleanup, use_llm)

def render_results(df: pd.DataFrame, brand_name: str = ""):
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
        st.info(f"Showing first 100 rows of {len(df):,} total rows.")
    
    try:
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        
        # Generate filename with brand name or default
        if brand_name:
            filename = f"{brand_name.replace(' ', '')}_TaxonomyGuardianCleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        else:
            filename = f"TaxonomyGuardianCleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        st.download_button(
            "📥 Download Cleaned File (.xlsx)",
            data=output_buffer.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")

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

def main():
    try:
        brand_df = load_brand_reference(show_upload=True)
        
        (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
         selected_brand, log_changes_only, max_logs, run_cleanup, use_llm) = render_sidebar_controls(brand_df)
        
        st.header("🛡️ Data Cleanup Status")
        
        if uploaded_file is None:
            st.info("Upload a data file in the sidebar to get started")
            
            with st.expander("How to Use Taxonomy Guardian", expanded=True):
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
            st.error("Failed to load the uploaded file.")
            render_logs()
            return
        
        st.success(f"File loaded: **{uploaded_file.name}**")
        
        file_col1, file_col2 = st.columns(2)
        with file_col1:
            st.metric("Rows", f"{len(raw_df):,}")
        with file_col2:
            st.metric("Columns", len(raw_df.columns))
        
        try:
            processed_df = normalize_column_headers(raw_df)
            
            if processed_df.empty:
                st.error("File appears to be empty after processing.")
                render_logs()
                return
            
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_df.columns]
            
            if missing_columns:
                st.warning(f"Missing required columns (added as empty): {', '.join(missing_columns)}")
            else:
                st.success("All required columns are present")
            
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
                    st.warning("Some columns have significant missing data. Check the logs for details.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            log_event("ERROR", f"File processing error: {str(e)}")
            render_logs()
            return
        
        st.subheader(f"Selected Cleanup: {cleanup_type}")
        
        if cleanup_type == "Brand Accuracy":
            if not selected_manufacturer or not selected_brand:
                st.warning("Please select both manufacturer and brand in the sidebar")
                render_logs()
                return
            else:
                st.success(f"Target: **{selected_manufacturer}** → **{selected_brand}**")
        
        if not run_cleanup:
            st.info("Click **Run Cleanup** in the sidebar to process your data")
            render_logs()
            return
        
        try:
            log_event("INFO", f"Starting {cleanup_type} cleanup", 
                      rows=len(processed_df),
                      cleanup_type=cleanup_type)
            
            with st.spinner(f"Running {cleanup_type.lower()}..."):
                
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
                    st.error("Unknown cleanup type selected")
                    return
            
            st.success("Cleanup completed successfully!")
            log_event("SUCCESS", f"{cleanup_type} cleanup completed successfully")
            
            render_results(cleaned_df, selected_brand if cleanup_type == "Brand Accuracy" else "")
        
        except Exception as e:
            error_msg = f"Error during {cleanup_type.lower()}: {str(e)}"
            st.error(error_msg)
            log_event("ERROR", error_msg)
        
        render_logs()
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        log_event("ERROR", f"Application error: {str(e)}")
        render_logs()

if __name__ == "__main__":
    main()
