import streamlit as st
from huggingface_hub import InferenceClient

# Hugging Face model (runs on cloud, not your local machine)
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=st.secrets["HF_TOKEN"])

# Page setup
st.set_page_config(page_title="Customer Message Classifier", layout="centered")
st.title("🧠 Customer Message Classifier (LLM-Powered)")
st.markdown("Classify customer support messages into one of 35 banking categories using a powerful LLM.")

# Input box
msg = st.text_area("📩 Enter a customer message below:", height=150)

# Function to build the prompt (based on your code)
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

[CATEGORY LIST OMITTED HERE TO KEEP THE PROMPT SHORT FOR API]

---

### EXAMPLES:

Message: "What the hell is going on with my loan?"
Answer: Frustrated/Negative Tone (Unclear Context)

Message: "Loan approved last week but no disbursement yet"
Answer: Disbursement Delay

Message: "You deducted EMI even after loan closure"
Answer: Foreclosure Refund/Extra EMI Refund

Message: "When do I get my documents back?"
Answer: Document Return Query

Message: "Need escalation matrix or contact of nodal officer"
Answer: Escalation/Grievance Request

Message: "Login alert from unknown device – not me"
Answer: Duplicate Login Alert/Security Concern

Message: "Thank you so much for resolving this!"
Answer: Appreciation/Positive Feedback

Message: "{msg}"
Answer:"""

# LLM call
def classify_with_llm(msg):
    prompt = build_prompt(msg)
    response = client.text_generation(prompt, max_new_tokens=30, temperature=0.3, stop_sequences=["\n"])
    return response.strip()

# Submit button
if st.button("🚀 Classify"):
    with st.spinner("Classifying with Phi-3..."):
        category = classify_with_llm(msg)
        st.success(f"**Predicted Category:** {category}")
