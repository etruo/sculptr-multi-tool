import os
import tempfile
import json
from pathlib import Path
from io import BytesIO
from datetime import datetime
import openpyxl

import streamlit as st
from openai import OpenAI

import extractor
import build_model

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Page configuration
st.set_page_config(
    page_title="🏢 Motherfucking (MF) Underwriting Helper",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏢  Value-Add Multifamily Parser + Model Filler")

# ---------- sidebar: API key, settings & history ------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 OpenAI API key")
    if "OPENAI_API_KEY" not in os.environ:
        user_key = st.text_input("Paste your key", type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            st.success("Key stored for this session")
    
    st.markdown("### ⚙️ Settings")
    debug_mode = st.checkbox("Enable debug mode", value=False, help="Show detailed processing logs")
    if debug_mode:
        os.environ["DEBUG"] = "1"
    else:
        os.environ.pop("DEBUG", None)
    
    # Display history
    if st.session_state.history:
        st.markdown("### 📚 Previous Models")
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"#{idx + 1}: {item['property_name']} ({item['date']})"):
                st.json(item['data'])
                if 'excel_data' in item:
                    st.download_button(
                        "📥 Download This Model",
                        data=item['excel_data'],
                        file_name=item['filename'],
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

# ---------- input choice ----------------------------------------------------
mode = st.radio("Choose input type", ("E-mail text", "Offering Memorandum PDF"))
txt, pdf = "", None

if mode == "E-mail text":
    txt = st.text_area("✉️ Paste e-mail body", height=250)
else:
    pdf = st.file_uploader("📄 Upload OM (PDF)", type=["pdf"])
    if pdf:
        st.info("Using multi-template-v2.xlsx as the base template")

run = st.button("🚀 Extract + Build", disabled=(not txt and not pdf))

if run:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with st.spinner("Extracting data and building model..."):
            # 1) extract -----------------------------------------------------------------
            if mode == "E-mail text":
                fields = extractor.parse_deal(plain_text=txt)
                src_name = "email"
            else:
                tmp = Path(tempfile.gettempdir()) / "upload.pdf"
                tmp.write_bytes(pdf.getvalue())
                fields = extractor.parse_deal(pdf_path=tmp)
                src_name = Path(pdf.name).stem

            # Get property name from JSON data
            property_name = fields.get("Property Name", "Unnamed Property")

            # Show extraction results in an expander
            with st.expander("🔍 Extracted Data", expanded=True):
                st.json(fields)

            # 2) build model -------------------------------------------------------------
            template_path = Path(__file__).with_name("multi-template-v2.xlsx")
            if not template_path.exists():
                st.error(f"Template file not found: {template_path}")
                st.stop()

            output_path = Path(tempfile.gettempdir()) / f"{src_name}_underwriting.xlsx"
            
            if debug_mode:
                st.write("Debug Information:")
                st.write({
                    "Template": str(template_path),
                    "Output": str(output_path),
                    "Source": src_name
                })

            wb_path = build_model._populate_workbook(
                template=template_path,
                out=output_path,
                data=fields,
            )

            # Format current date
            current_date = datetime.now().strftime("%m.%d.%y")
            
            # Create standardized filename
            standardized_filename = f"2-page {property_name} Automated {current_date}.xlsx"
            
            if debug_mode:
                st.write("File Information:")
                st.write({
                    "Property Name": property_name,
                    "Date": current_date,
                    "Final Filename": standardized_filename
                })

            # --------------------------- download section -----------------------------------
            with open(wb_path, "rb") as fh:
                bytes_xlsx = fh.read()

            st.success("✅ Model successfully built!")
            
            # Add to session history
            st.session_state.history.append({
                'property_name': property_name,
                'date': current_date,
                'data': fields,
                'excel_data': bytes_xlsx,
                'filename': standardized_filename
            })

            # Create download button
            st.download_button(
                "📥 Download Excel Model",
                data=bytes_xlsx,
                file_name=standardized_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        if debug_mode:
            st.exception(e) 