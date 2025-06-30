import streamlit as st
import requests

st.set_page_config(page_title="Customer Message Classifier", layout="centered")
st.title("📨 Customer Message Classifier")
st.write("Classify incoming customer messages into a relevant category using a free LLM.")

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

def build_prompt(msg):
    return f"""
You are a highly trained AI support assistant working for a large banking customer support center. Customers may contact via email, chatbot, mobile app, or social media. Messages may be short, vague, appreciative, angry, confused, or demanding.

Your job is to classify **each customer message into exactly ONE most relevant category** from the list below.

--- Rules ---

- Emotionally charged but unclear → **Frustrated/Negative Tone (Unclear Context)**
- Angry and clearly about delay, app, branch, no response → **Customer Complaint**
- Mentions branch manager, cleanliness, staff behavior → **Branch/Staff/Service Feedback**
- Praising service or agent → **Appreciation/Positive Feedback**
- Says "escalation", "escalation matrix", wants senior support → **Escalation/Grievance Request**
- Documents submitted + asks when they’ll be returned → **Document Return Query**
- Says loan closed but EMI still deducted → **Foreclosure Refund/Extra EMI Refund**
- Mentions login alerts from new device → **Duplicate Login Alert/Security Concern**
- KYC expired, face mismatch, update needed → **KYC Expiry/Update**
- Nothing fits → **Unclassified**

--- CATEGORIES ---

Loan Application  
Loan Eligibility  
Disbursement Delay  
Prepayment/Part Payment/Foreclosure  
Foreclosure Refund/Extra EMI Refund  
Interest Rate Query  
EMI Bounce  
EMI Schedule Change  
EMI Overcharge/Error  
Loan Top-Up Request  
Balance Transfer  
Loan Rejection  
CIBIL/Score Concern  
Document Submission  
Document Return Query  
KYC Expiry/Update  
Statement Request  
NOC/Loan Closure  
Account Statement Request  
Address/Mobile Update  
Login/Netbanking Issue  
Duplicate Login Alert/Security Concern  
General Inquiry  
Customer Complaint  
Escalation/Grievance Request  
Technical Issue  
Branch Visit Request  
Branch/Staff/Service Feedback  
Appreciation/Positive Feedback  
Registration  
Contact Assistance  
Card Lost/Block Request  
Social Media Escalation  
Frustrated/Negative Tone (Unclear Context)  
Unclassified

--- EXAMPLES ---

Message: "What the hell is going on with my loan?"  
Answer: Frustrated/Negative Tone (Unclear Context)

Message: "Loan approved last week but no disbursement yet"  
Answer: Disbursement Delay

Message: "{msg}"  
Answer:"""

def classify_with_llm(msg):
    prompt = build_prompt(msg)
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 30,
            "temperature": 0.3,
            "stop": ["\n"]
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            text = result[0]["generated_text"]
            answer = text.split("Answer:")[-1].strip()
            return answer
        else:
            return "❌ Unexpected output format from model."

    except Exception as e:
        return f"❌ Error: {str(e)}"

# UI
msg = st.text_area("✉️ Enter customer message here:")

if st.button("Classify Message"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Classifying..."):
            category = classify_with_llm(msg)
        st.success(f"🔖 Predicted Category:\n\n**{category}**")
