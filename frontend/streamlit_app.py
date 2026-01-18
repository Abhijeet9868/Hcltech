"""
Form Extraction UI - Streamlit Frontend
Self-service portal for uploading and extracting data from banking forms.
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import io

# Configuration
API_URL = "http://localhost:5000/api"

# Page configuration
st.set_page_config(
    page_title="Form Data Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.25rem 0;
    }
    
    /* Field display */
    .field-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .field-label {
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .field-value {
        color: white;
        font-size: 1.1rem;
        margin-top: 0.25rem;
    }
    
    /* Confidence indicator */
    .confidence-high { color: #38ef7d; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #f5576c; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def get_confidence_class(confidence):
    """Get CSS class for confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    return "confidence-low"


def get_confidence_emoji(confidence):
    """Get emoji for confidence level."""
    if confidence >= 0.8:
        return "🟢"
    elif confidence >= 0.6:
        return "🟡"
    return "🔴"


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Form Data Extractor</h1>
        <p>AI-powered information extraction from banking forms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select a page:",
            ["🏠 Extract New", "📊 History", "📈 Statistics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool extracts structured data from scanned banking application forms using:
        - 🔍 **OCR** for text extraction
        - 🧠 **AI** for intelligent field detection
        - ✅ **Validation** for data accuracy
        """)
    
    if "🏠 Extract New" in page:
        show_extraction_page()
    elif "📊 History" in page:
        show_history_page()
    elif "📈 Statistics" in page:
        show_stats_page()


def show_extraction_page():
    """Main extraction page."""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Form")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Supported formats: PNG, JPG, JPEG, PDF"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Form", use_container_width=True)
            
            if st.button("🚀 Extract Data", use_container_width=True):
                with st.spinner("Processing form..."):
                    try:
                        # Call API
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                        response = requests.post(f"{API_URL}/extract", files=files, timeout=120)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['extraction_result'] = result
                            st.success("✅ Extraction completed!")
                        else:
                            st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Cannot connect to API. Make sure the Flask backend is running.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("### 📋 Extracted Data")
        
        if 'extraction_result' in st.session_state:
            result = st.session_state['extraction_result']
            show_extraction_result(result)
        else:
            st.info("Upload a form and click 'Extract Data' to see results here.")


def show_extraction_result(result):
    """Display extraction results."""
    
    # Overall stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_conf = result.get('confidence_scores', {}).get('overall', 0)
        st.metric(
            "Overall Confidence",
            f"{overall_conf*100:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Form Type",
            result.get('form_type', 'Unknown').replace('_', ' ').title()
        )
    
    with col3:
        processing_time = result.get('processing_time_ms', 0)
        st.metric(
            "Processing Time",
            f"{processing_time}ms"
        )
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Fields", "⚠️ Review Needed", "🔧 Raw Data"])
    
    with tab1:
        fields = result.get('fields', {})
        confidence_scores = result.get('confidence_scores', {})
        validation_results = result.get('validation_results', {})
        
        if not fields:
            st.warning("No fields extracted from this document.")
            st.info("This could be due to:")
            st.markdown("""
            - Low quality scan
            - Unsupported form format  
            - Handwritten text that couldn't be recognized
            """)
        else:
            # Display fields
            for field_name, value in fields.items():
                if value is None or str(value).strip() == '':
                    continue
                
                conf = confidence_scores.get(field_name, 0)
                validation = validation_results.get(field_name, {})
                is_valid = validation.get('is_valid', True)
                errors = validation.get('errors', [])
                
                emoji = get_confidence_emoji(conf)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    display_name = field_name.replace('_', ' ').title()
                    st.markdown(f"""
                    <div class="field-card">
                        <div class="field-label">{display_name}</div>
                        <div class="field-value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"{emoji} **{conf*100:.0f}%**")
                    if not is_valid:
                        for error in errors:
                            st.caption(f"⚠️ {error}")
    
    with tab2:
        needs_review = result.get('needs_review', [])
        cross_warnings = result.get('cross_validation_warnings', [])
        doc_id = result.get('_id', None)
        
        if needs_review:
            st.warning(f"**{len(needs_review)} fields need review:**")
            
            # Create a form for editing
            edited_fields = {}
            for field in needs_review:
                value = result.get('fields', {}).get(field, '')
                conf = result.get('confidence_scores', {}).get(field, 0)
                emoji = get_confidence_emoji(conf)
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    new_value = st.text_input(
                        f"{emoji} {field.replace('_', ' ').title()}",
                        value=str(value) if value else '',
                        key=f"review_{field}_{doc_id}"
                    )
                    edited_fields[field] = new_value
                with col2:
                    st.markdown(f"<br>**{conf*100:.0f}%**", unsafe_allow_html=True)
            
            # Save button
            if doc_id and st.button("💾 Save Corrections", use_container_width=True, key=f"save_{doc_id}"):
                try:
                    # Update the fields in the database
                    update_data = {
                        'fields': {**result.get('fields', {}), **edited_fields},
                        'status': 'reviewed'
                    }
                    response = requests.put(
                        f"{API_URL}/results/{doc_id}",
                        json=update_data,
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.success("✅ Corrections saved successfully!")
                        # Update session state if it exists
                        if 'extraction_result' in st.session_state:
                            st.session_state['extraction_result']['fields'].update(edited_fields)
                            st.session_state['extraction_result']['status'] = 'reviewed'
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to save: {response.json().get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error saving: {str(e)}")
        else:
            fields = result.get('fields', {})
            if not fields:
                st.info("No fields were extracted to review.")
                # Show OCR text if available
                ocr_text = result.get('ocr_text', '')
                if ocr_text:
                    st.markdown("### 📝 Raw OCR Text")
                    st.text_area("Extracted Text", ocr_text, height=200, disabled=True)
            else:
                st.success("✅ All fields validated successfully!")
        
        if cross_warnings:
            st.markdown("### Cross-Validation Warnings")
            for warning in cross_warnings:
                st.warning(warning)
    
    with tab3:
        st.json(result)
    
    # Export buttons
    st.markdown("---")
    st.markdown("### 📥 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            "📄 Download JSON",
            data=json_str,
            file_name=f"extraction_{result.get('_id', 'result')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create CSV
        fields = result.get('fields', {})
        df = pd.DataFrame([
            {
                'Field': k.replace('_', ' ').title(),
                'Value': v,
                'Confidence': result.get('confidence_scores', {}).get(k, 'N/A'),
                'Valid': result.get('validation_results', {}).get(k, {}).get('is_valid', 'N/A')
            }
            for k, v in fields.items() if v
        ])
        
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                data=csv,
                file_name=f"extraction_{result.get('_id', 'result')}.csv",
                mime="text/csv",
                use_container_width=True
            )


def show_history_page():
    """Show extraction history."""
    st.markdown("### 📊 Extraction History")
    
    try:
        response = requests.get(f"{API_URL}/history", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                st.info("No extractions yet. Upload a form to get started!")
                return
            
            # Display as table
            df = pd.DataFrame([
                {
                    'ID': r.get('_id', '')[:8] + '...',
                    'Filename': r.get('filename', 'N/A'),
                    'Form Type': r.get('form_type', 'N/A').replace('_', ' ').title(),
                    'Date': r.get('upload_date', '')[:10],
                    'Status': r.get('status', 'N/A'),
                    'Confidence': f"{r.get('confidence_scores', {}).get('overall', 0)*100:.0f}%"
                }
                for r in results
            ])
            
            st.dataframe(df, use_container_width=True)
            
            # Detail view
            st.markdown("---")
            st.markdown("### View Details")
            
            selected_id = st.selectbox(
                "Select an extraction to view:",
                options=[r.get('_id', '') for r in results],
                format_func=lambda x: f"{x[:8]}... - {next((r.get('filename', 'N/A') for r in results if r.get('_id') == x), 'N/A')}"
            )
            
            if selected_id:
                selected = next((r for r in results if r.get('_id') == selected_id), None)
                if selected:
                    show_extraction_result(selected)
        else:
            st.error("Failed to fetch history")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the Flask backend is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_stats_page():
    """Show statistics."""
    st.markdown("### 📈 Extraction Statistics")
    
    try:
        response = requests.get(f"{API_URL}/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Extractions", stats.get('total_extractions', 0))
            
            with col2:
                st.metric("Needs Review", stats.get('needs_review_count', 0))
            
            with col3:
                form_types = stats.get('form_types', {})
                st.metric("Form Types", len(form_types))
            
            # Form types breakdown
            if form_types:
                st.markdown("---")
                st.markdown("### Form Types Distribution")
                df = pd.DataFrame([
                    {'Form Type': k.replace('_', ' ').title(), 'Count': v}
                    for k, v in form_types.items()
                ])
                st.bar_chart(df.set_index('Form Type'))
        else:
            st.error("Failed to fetch statistics")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the Flask backend is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
