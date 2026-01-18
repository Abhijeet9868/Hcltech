# Information Extraction from Scanned User-Filled Forms

An AI-powered system for extracting structured data from scanned banking application forms using OCR, layout detection, and LLM-based field extraction.

## 🚀 Features

- **OCR Engine**: Tesseract-based text extraction with confidence scoring
- **Layout Detection**: Intelligent key-value pair identification for banking forms
- **LLM Extraction**: OpenAI GPT-4 Vision for accurate field extraction
- **Validation**: Multi-layer validation with format checking and cross-field consistency
- **Modern UI**: Streamlit-based self-service portal with drag-and-drop upload
- **Database**: MongoDB for storing extraction history and results

## 📁 Project Structure

```
hcl-tech-matrix AI/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── config.py               # Configuration settings
│   ├── requirements.txt        # Backend dependencies
│   ├── modules/
│   │   ├── preprocessing.py    # Image deskew, denoise, enhance
│   │   ├── ocr_engine.py       # Tesseract OCR integration
│   │   ├── layout_detector.py  # Key-value pair detection
│   │   ├── llm_extractor.py    # OpenAI GPT-4 Vision
│   │   └── validator.py        # Field validation
│   └── database/
│       └── mongo_client.py     # MongoDB operations
├── frontend/
│   ├── streamlit_app.py        # Streamlit UI
│   └── requirements.txt        # Frontend dependencies
└── sample_forms/               # Test form images
```

## 🛠️ Installation

### Prerequisites

1. **Python 3.11+**
2. **Tesseract OCR**: [Download](https://github.com/UB-Mannheim/tesseract/wiki)
3. **MongoDB**: [Download](https://www.mongodb.com/try/download/community)
4. **OpenAI API Key**: [Get one](https://platform.openai.com/api-keys)

### Setup

1. **Clone and navigate to project:**
   ```bash
   cd "hcl-tech-matrix AI"
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   cd ../frontend
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cd ../backend
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

5. **Update Tesseract path in `.env`:**
   ```
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

## 🚀 Running the Application

### 1. Start MongoDB
```bash
mongod
```

### 2. Start Flask Backend
```bash
cd backend
python app.py
```
Server runs at: `http://localhost:5000`

### 3. Start Streamlit Frontend
```bash
cd frontend
streamlit run streamlit_app.py
```
UI runs at: `http://localhost:8501`

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload form image |
| `/api/extract` | POST | Extract data from form |
| `/api/results/<id>` | GET | Get extraction by ID |
| `/api/history` | GET | Get all extractions |
| `/api/export/<id>/<format>` | GET | Export as JSON/CSV |
| `/api/stats` | GET | Get statistics |

## 🔍 Extracted Fields (Banking Forms)

- Personal: Name, DOB, Gender, Nationality, Father's Name
- Address: Permanent/Current Address, City, State, Pincode
- Contact: Phone, Mobile, Email
- Identity: PAN, Aadhaar, Passport
- Employment: Employer, Occupation, Income
- Banking: Account Type, Nominee, Branch

## ✅ Validation Rules

| Field | Format | Example |
|-------|--------|---------|
| PAN | ABCDE1234F | BNZPM1234K |
| Aadhaar | 12 digits | 1234 5678 9012 |
| Phone | 10-12 digits | +91 9876543210 |
| Email | standard@format.com | user@email.com |
| Pincode | 6 digits | 560001 |
| Date | ISO format | 1992-07-14 |

## 📊 Confidence Levels

- 🟢 **High (>80%)**: Reliable extraction
- 🟡 **Medium (60-80%)**: May need review
- 🔴 **Low (<60%)**: Needs manual verification

## 🧪 Testing

```bash
# Test backend health
curl http://localhost:5000/api/health

# Test extraction
curl -X POST -F "file=@test_form.jpg" http://localhost:5000/api/extract
```

## 📝 License

MIT License

## 👥 Authors

HCL Tech Matrix AI Team
