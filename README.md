# MF Underwriting Helper

A Streamlit application that helps extract data from multifamily property offering memorandums and automatically populates an Excel underwriting model.

## Features

- Extract data from PDF offering memorandums or email text
- Automatically populate standardized Excel model
- Track history of processed deals
- Debug mode for detailed processing information
- Downloadable Excel models with standardized naming

## Required Files

Make sure these files are present in your deployment:

- `template_app_v1.py` - Main Streamlit application
- `build_model.py` - Excel population logic
- `extractor.py` - Data extraction logic
- `multi-template-v2.xlsx` - Excel template file

## Deployment

1. Create a Streamlit account at https://streamlit.io/
2. Connect your GitHub repository
3. Add your OpenAI API key to Streamlit secrets
4. Deploy!
