# streamlit_app.py
# Taxonomy Guardian - Final clean version without variable scope issues

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
    """Normalize column headers to match required format - FIXED VERSION"""
    df = df.copy()
    
    # Debug: log original column structure
    log_event("INFO", "File column analysis", 
              original_columns=df.columns.tolist()[:20],
              total_columns=len(df.columns),
              sample_data=df.iloc[0].tolist()[:10] if not df.empty else [])
    
    # Remove completely empty columns first
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.columns.notna()]  # Remove columns with NaN names
    
    # Remove unnamed columns (Excel sometimes adds these)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    log_event("INFO", "After cleaning empty columns", 
              cleaned_columns=df.columns.tolist(),
              remaining_rows=len(df))
    
    # Create mapping from incoming columns to required columns
    column_mapping = {}
    normalized_incoming = {col: col.strip().upper() for col in df.columns}
    
    for incoming_col, normalized_col in normalized_incoming.items():
        if normalized_col in REQUIRED_COLUMNS:
            column_mapping[incoming_col] = normalized_col
    
    # Rename columns
    if column_mapping:
        df = df.rename(columns=column_mapping)
        log_event("INFO", "Column mapping applied", mapping=column_mapping)
    
    # Add missing required columns as empty strings
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    return df

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            if col in ["Correct Brand?", "Correct Categories?", "Vague Description?"]:
                df[col] = "N"
            else:
                df[col] = ""
    return df

# =========================
# File I/O Functions
# =========================
@st.cache_data(show_spinner=False)
def read_excel_file(file_path_or_buffer) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(file_path_or_buffer)
        df.columns = [col.strip().upper() for col in df.columns]
        
        string_columns = ["BRAND", "MANUFACTURER", "WEBSITE", "ALLOWED_PRODUCT_TYPES"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        log_event("ERROR", f"Failed to read Excel file: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def read_user_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO, StringIO
    
    try:
        file_name = file_name.lower()
        
        if file_name.endswith('.csv'):
            content = file_bytes.decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(content))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
        
        log_event("INFO", f"Successfully loaded file: {file_name}", rows=len(df), columns=len(df.columns))
        return df
        
    except Exception as e:
        log_event("ERROR", f"Failed to read user file: {str(e)}")
        raise

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
# Brand Processing Functions
# =========================
def extract_brand_from_description(description: str, master_brands: set) -> Optional[str]:
    """Extract brand from description"""
    desc_lower = description.lower()
    
    # Look for any master brand that appears in the description
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
    max_logs: int = 200
) -> pd.DataFrame:
    
    if not selected_manufacturer or not selected_brand:
        log_event("ERROR", "No manufacturer or brand selected")
        return df
    
    log_event("INFO", "Starting brand accuracy cleanup", 
              manufacturer=selected_manufacturer, 
              brand=selected_brand, 
              rows=len(df))
    
    df = df.copy()
    df = ensure_output_columns(df)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get brand information - SIMPLE VERSION
    master_brands = get_master_brand_set(brand_df)
    allowed_types = get_allowed_product_types(brand_df, selected_brand)
    selected_brand_lower = selected_brand.lower()
    
    # Debug logging
    log_event("INFO", "Brand matching setup", 
              selected_brand=selected_brand,
              allowed_types=allowed_types)
    
    df["Correct Brand?"] = "N"
    df["Suggested Brand"] = ""
    df["Match Strength"] = ""
    
    changes_logged = 0
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing row {idx + 1} of {total_rows}")
        
        description = safe_str(row.get("DESCRIPTION", ""))
        desc_lower = description.lower()
        
        # Debug logging for first few rows to see what data we're getting
        if idx < 3:
            log_event("INFO", f"Row {idx + 1} debug", 
                      description=description,
                      description_length=len(description),
                      all_columns=list(row.keys())[:10])  # Show first 10 column names
        
        # ENHANCED MATCHING: Check if description contains any allowed product type or variations
        belongs_to_selected = False
        for product_type in allowed_types:
            pt_lower = product_type.lower().strip()
            
            # Direct match
            if pt_lower in desc_lower:
                belongs_to_selected = True
                break
            
            # Add basic variations for common patterns
            if pt_lower == "chips":
                if any(term in desc_lower for term in ["chip", "chips", "crisps"]):
                    belongs_to_selected = True
                    break
            elif pt_lower == "veggie chips":
                if any(term in desc_lower for term in ["veggie chip", "vegetable chip", "beet chip", "sweet potato chip"]):
                    belongs_to_selected = True
                    break
            elif pt_lower == "snacks":
                if any(term in desc_lower for term in ["snack", "snacks"]):
                    belongs_to_selected = True
                    break
            # Add powder/supplement matching for Terra Kai products
            elif "powder" in pt_lower or "supplement" in pt_lower:
                if any(term in desc_lower for term in ["powder", "supplement"]):
                    belongs_to_selected = True
                    break
        
        # Check for wine conflicts (specific to chips brands)
        if belongs_to_selected and any('chip' in pt.lower() for pt in allowed_types):
            wine_indicators = ['ml', '750', 'cabernet', 'merlot', 'sauvignon', 'wine']
            if any(indicator in desc_lower for indicator in wine_indicators):
                belongs_to_selected = False
        
        # Mark as correct or incorrect
        if belongs_to_selected:
            df.at[idx, "Correct Brand?"] = "Y"
            df.at[idx, "Match Strength"] = "High"
            df.at[idx, "Suggested Brand"] = ""
            continue
        
        # Product doesn't belong - suggest different brand
        df.at[idx, "Correct Brand?"] = "N"
        
        # Try to extract brand from description
        suggested_brand = extract_brand_from_description(description, master_brands)
        
        # Never suggest the same brand for N items
        if suggested_brand and suggested_brand.lower() == selected_brand_lower:
            suggested_brand = None
        
        # Final fallback
        if not suggested_brand:
            if any(wine_ind in desc_lower for wine_ind in ['ml', '750', 'wine', 'cabernet']):
                # Look for specific wine Terra brands
                if 'terra valentine' in desc_lower or 'valentine' in desc_lower:
                    suggested_brand = "Terra D'Oro"
                elif 'terra blanca' in desc_lower or 'blanca' in desc_lower:
                    suggested_brand = "Terra Blanca"
                elif 'terra vega' in desc_lower or 'vega' in desc_lower:
                    suggested_brand = "Terra Wine"
                else:
                    suggested_brand = "Terra Wine"  # Generic wine Terra brand
            else:
                suggested_brand = "Unknown Brand"
        
        df.at[idx, "Suggested Brand"] = suggested_brand
        df.at[idx, "Match Strength"] = "Low"
        
        # Log changes
        if log_changes_only and changes_logged < max_logs:
            log_event("INFO", "Brand correction needed",
                      fido=safe_str(row.get("FIDO")),
                      description=description[:100],
                      suggested_brand=suggested_brand)
            changes_logged += 1
    
    progress_bar.progress(1.0)
    status_text.text("Brand cleanup completed!")
    
    correct_count = len(df[df["Correct Brand?"] == "Y"])
    incorrect_count = len(df[df["Correct Brand?"] == "N"])
    
    log_event("INFO", "Brand cleanup completed",
              correct_brands=correct_count,
              incorrect_brands=incorrect_count,
              total_rows=len(df))
    
    return df

def category_hierarchy_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    log_event("INFO", "Starting category hierarchy cleanup", rows=len(df))
    
    df = df.copy()
    df = ensure_output_columns(df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)
    
    for idx, row in df.iterrows():
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
    
    progress_bar.progress(1.0)
    status_text.text("Category cleanup completed!")
    
    correct_count = len(df[df["Correct Categories?"] == "Y"])
    log_event("INFO", "Category cleanup completed", 
              correct_categories=correct_count,
              total_rows=len(df))
    
    return df

def vague_description_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    log_event("INFO", "Starting description cleanup", rows=len(df))
    
    df = df.copy()
    df = ensure_output_columns(df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)
    
    vague_pattern = r'\b(' + '|'.join(VAGUE_TERMS) + r')\b'
    
    for idx, row in df.iterrows():
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Analyzing description for row {idx + 1} of {total_rows}")
        
        description = safe_str(row.get("DESCRIPTION", ""))
        
        is_vague = bool(re.search(vague_pattern, description.lower())) or len(description.strip()) < 10
        
        df.at[idx, "Vague Description?"] = "Y" if is_vague else "N"
        
        if is_vague:
            df.at[idx, "Match Strength"] = "Medium"
    
    progress_bar.progress(1.0)
    status_text.text("Description cleanup completed!")
    
    vague_count = len(df[df["Vague Description?"] == "Y"])
    log_event("INFO", "Description cleanup completed",
              vague_descriptions=vague_count,
              total_rows=len(df))
    
    return df

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
            file_key = f"{uploaded_file.name}:{hash(file_bytes)}"
            
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
            
            manufacturer_options = top_manufacturers if show_top_only else sorted(brands_by_manufacturer.keys())
            
            selected_manufacturer = st.sidebar.selectbox(
                "Select manufacturer:",
                options=manufacturer_options,
                help="Choose the manufacturer to focus cleanup on"
            )
            
            if selected_manufacturer:
                brand_options = brands_by_manufacturer.get(selected_manufacturer, [])
                selected_brand = st.sidebar.selectbox(
                    "Select brand:",
                    options=brand_options,
                    help="Choose the specific brand for cleanup"
                )
        else:
            st.sidebar.warning("⚠️ No brand reference data available.")
    
    st.sidebar.header("⚙️ Advanced Settings")
    
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
    
    # Add clear form button
    clear_form = st.sidebar.button(
        "🗑️ Clear Form", 
        use_container_width=True,
        help="Clear uploaded file and reset form"
    )
    
    if clear_form:
        # Clear session state
        if "_last_file_key" in st.session_state:
            del st.session_state["_last_file_key"]
        if "_raw_df" in st.session_state:
            del st.session_state["_raw_df"]
        st.rerun()
    
    return (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
            selected_brand, log_changes_only, max_logs, run_cleanup)

def render_results(df: pd.DataFrame):
    st.subheader("📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    
    with col2:
        if "Correct Brand?" in df.columns:
            correct_brands = len(df[df["Correct Brand?"] == "Y"])
            st.metric("Correct Brands", correct_brands)
    
    with col3:
        if "Correct Categories?" in df.columns:
            correct_categories = len(df[df["Correct Categories?"] == "Y"])
            st.metric("Correct Categories", correct_categories)
    
    with col4:
        if "Suggested Brand" in df.columns:
            suggestions = len(df[df["Suggested Brand"] != ""])
            st.metric("Suggestions Made", suggestions)
    
    st.dataframe(df.head(100), use_container_width=True)
    
    # Download functionality
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
    brand_df = load_brand_reference(show_upload=True)
    
    (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
     selected_brand, log_changes_only, max_logs, run_cleanup) = render_sidebar_controls(brand_df)
    
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
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_df.columns]
        
        if missing_columns:
            st.warning(f"⚠️ Missing required columns (will be added as empty): {', '.join(missing_columns)}")
        else:
            st.success("✅ All required columns are present")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
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
                    max_logs
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

if __name__ == "__main__":
    main()
