import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="Taxonomy_Guardian",
    page_icon="Taxonomy_Guardian.png️",
    layout="wide"
)

st.image("Taxonomy_Guardian.png", width=150)
st.markdown("#### Built for Implementation, Sales & RQ")


st.title("Taxonomy_Guardian")
st.markdown("Ensure brand, category, and description accuracy based on Fetch's taxonomy standards.")

# --- Sidebar Inputs ---
st.sidebar.header("Step 1: Upload File")
uploaded_file = st.sidebar.file_uploader("Upload your Snowflake export (CSV or Excel)", type=["csv", "xlsx"])

st.sidebar.header("Step 2: Select Cleanup Options")
cleanup_options = st.sidebar.multiselect(
    "Which cleanup actions should we perform?",
    ["Brand Accuracy", "Category Hierarchy Cleanup", "Vague Description Cleanup"]
)

brand_required = "Brand Accuracy" in cleanup_options

if brand_required:
    st.sidebar.markdown("#### Brand Accuracy Settings")
    target_brand = st.sidebar.text_input("Target Brand (required)")
    target_manufacturer = st.sidebar.text_input("Target Manufacturer (required)")

    if not target_brand or not target_manufacturer:
        st.warning("Please enter both Target Brand and Target Manufacturer for Brand Accuracy cleanup.")

# --- Helper Logic Placeholders ---

def clean_brand_accuracy(df, brand, manufacturer):
    df["Correct Brand?"] = "Y"
    df["Suggested Brand"] = ""

    # Dummy placeholder logic – real version should reference external brand sources
    for i, row in df.iterrows():
        description = str(row.get("DESCRIPTION", "")).lower()
        current_brand = str(row.get("BRAND", "")).lower()
        if brand.lower() not in description:
            df.at[i, "Correct Brand?"] = "N"
            df.at[i, "Suggested Brand"] = "Pepsi"  # Replace with actual logic

    return df


def clean_category_hierarchy(df):
    df["Correct Categories?"] = "Y"
    df["Suggested Category 1"] = ""
    df["Suggested Category 2"] = ""
    df["Suggested Category 3"] = ""

    # Dummy placeholder logic
    for i, row in df.iterrows():
        category_1 = str(row.get("CATEGORY_1", "")).lower()
        description = str(row.get("DESCRIPTION", "")).lower()
        if "chips" in description and category_1 != "snacks":
            df.at[i, "Correct Categories?"] = "N"
            df.at[i, "Suggested Category 1"] = "Snacks"
            df.at[i, "Suggested Category 2"] = "Salty Snacks"
            df.at[i, "Suggested Category 3"] = "Chips"
    return df


def clean_vague_descriptions(df):
    df["Vague Description?"] = "N"
    df["Suggested New Description"] = ""

    for i, row in df.iterrows():
        desc = str(row.get("DESCRIPTION", "")).strip().lower()
        barcode = str(row.get("BARCODE", ""))
        if barcode.startswith(("31111", "51111")):
            continue
        if len(desc.split()) <= 2 or desc in {"item", "misc item", "product"}:
            df.at[i, "Vague Description?"] = "Y"
            df.at[i, "Suggested New Description"] = "Doritos Nacho Cheese 9.75oz"  # Replace with real logic
    return df

# --- Main Logic ---
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    original_df = df.copy()

    st.success(f"File uploaded. Rows: {len(df)}, Columns: {len(df.columns)}")

    # Perform selected cleanups
    if "Brand Accuracy" in cleanup_options and target_brand and target_manufacturer:
        df = clean_brand_accuracy(df, target_brand, target_manufacturer)

    if "Category Hierarchy Cleanup" in cleanup_options:
        df = clean_category_hierarchy(df)

    if "Vague Description Cleanup" in cleanup_options:
        df = clean_vague_descriptions(df)

    # --- Display Preview ---
    st.subheader("🔍 Preview Cleaned Data")
    st.dataframe(df.head(25), use_container_width=True)

    # --- Download Output ---
    st.subheader("📥 Download Cleaned File")
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)

    st.download_button(
        label="Download Excel File",
        data=towrite,
        file_name="taxonomy_guardian_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Awaiting file upload to begin.")

