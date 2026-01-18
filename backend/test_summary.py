from modules.llm_extractor import LLMExtractor
import os
from dotenv import load_dotenv

load_dotenv()

# Mock mock data
data = {
    "full_name": "John Doe",
    "account_type": "Savings Account",
    "date_of_birth": "1985-05-15",
    "nationality": "American",
    "phone": "555-0199",
    "email": "john.doe@example.com",
    "annual_income": "$75,000",
    "employer_name": "Acme Corp"
}

print("Testing summary generation with mock data...")
try:
    extractor = LLMExtractor()
    summary = extractor.generate_summary(data)
    print("\n--- Generated Summary ---")
    print(summary)
    print("-------------------------")
except Exception as e:
    print(f"\nError: {e}")
