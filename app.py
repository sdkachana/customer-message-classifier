import streamlit as st
import requests

st.set_page_config(page_title="Customer Message Classifier", layout="centered")

st.title("📨 Customer Message Classifier")
st.write("Classify incoming customer messages into a relevant category using a local LLM.")

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# Build the classification prompt
def build_prompt(msg):
    return f"""
You are a highly trained AI support assistant working for a large banking customer support center. Customers may contact via email, chatbot, mobile app, or social media. Messages may be short, vague, appreciative, angry, confused, or demanding.

Your job is to classify **each customer message into exactly ONE most relevant category** from the list below.

---

Use your judgment to apply these rules if necessary:

- If the message is emotionally charged but unclear → classify as **Frustrated/Negative Tone (Unclear Context)**
- If angry and clearly about delay, app, branch, no response → **Customer Complaint**
- If mentions branch manager, cleanliness, staff behavior → **Branch/Staff/Service Feedback**
- If praising service or agent → **Appreciation/Positive Feedback**
- If user says "need to escalate", "escalation matrix", or wants senior support → **Escalation/Grievance Request**
- If message says documents submitted and asks "when will they be returned?" → **Document Return Query**
- If mentions "loan closed but EMI still deducted" → **Foreclosure Refund/Extra EMI Refund**
- If mentions login alerts from new device or duplicate login attempt → **Duplicate Login Alert/Security Concern**
- If KYC expired, face mismatch, update required → **KYC Expiry/Update**
- If nothing applies → classify as **Unclassified**

---

### CATEGORIES (35 total):
- Loan Application
- Loan Eligibility
- Disbursement Delay
- Prepayment/Part Payment/Foreclosure
- Foreclosure Refund/Extra EMI Refund
- Interest Rate Query
- EMI Bounce
- EMI Schedule Change
- EMI Overcharge/Error
- Loan Top-Up Request
- Balance Transfer
- Loan Rejection
- CIBIL/Score Concern
- Document Submission
- Document Return Query
- KYC Expiry/Update
- Statement Request
- NOC/Loan Closure
- Account Statement Request
- Address/Mobile Update
- Login/Netbanking Issue
- Duplicate Login Alert/Security Concern
- General Inquiry
- Customer Complaint
- Escalation/Grievance Request
- Technical Issue
- Branch Visit Request
- Branch/Staff/Service Feedback
- Appreciation/Positive Feedback
- Registration
- Contact Assistance
- Card Lost/Block Request
- Social Media Escalation
- Frustrated/Negative Tone (Unclear Context)
- Unclassified

---

### EXAMPLES:

Message: "What the hell is going on with my loan?"
Answer: Frustrated/Negative Tone (Unclear Context)

Message: "Loan approved last week but no disbursement yet"
Answer: Disbursement Delay

Message: "{msg}"
Answer:"""

# Call Hugging Face API
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
            return "❌ Unable to classify. Model did not return expected output."

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
