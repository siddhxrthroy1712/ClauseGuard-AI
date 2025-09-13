import streamlit as st
import torch
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
from streamlit_chat import message

# Load API key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("Please set GOOGLE_API_KEY in Streamlit secrets or environment.")
    st.stop()

# Load Legal-BERT model
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("RahulMandadi/fine-tuned-legal-bert")
    tokenizer = BertTokenizer.from_pretrained("RahulMandadi/fine-tuned-legal-bert")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Class labels
labels = ["Cap on Liability", "Audit Rights", "Insurance", "None"]

# Legal-BERT classification
def classify_clause_legal_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return labels[prediction]

# Gemini risk analysis
def run_risk_analysis_gemini(clause):
    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        prompt = (
            f"You are a legal advisor. Analyze the contract clause '{clause}' and identify potential risks and mitigation strategies. "
            f"Provide a response in this format, with each point being 15-20 words for added depth:\n"
            f"Risk: [First risk in one sentence with no extra spacing, providing specific legal or practical concerns.]\n"
            f"Risk: [Second risk in one sentence with no extra spacing, highlighting a different legal or practical issue.]\n"
            f"Mitigation: [One mitigation in one sentence with no extra spacing, offering a clear and actionable solution.]\n"
            f"Ensure the risks and mitigation are directly relevant to the clause provided."
        )
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing clause: {str(e)}"

# Combined analysis
def classify_and_analyze_clause(clause):
    bert_label = classify_clause_legal_bert(clause)
    risk_analysis = run_risk_analysis_gemini(clause)
    response = (
        f"**Clause Analysis**\n\n"
        f"Input Clause: '{clause}'\n\n"
        f"**Classification**\n"
        f"Legal-BERT: {bert_label} (Accuracy: 97.87%)\n\n"
        f"**Risk Analysis (Gemini)**\n"
        f"{risk_analysis}"
    )
    return response

# Custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    /* Center title and instruction */
    h1 {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.2em;
    }
    p {
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    /* Style chat bubbles */
    .stChatMessage {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        max-width: 80%;
    }
    /* Style input box */
    .stTextInput > div > div > input {
        border: 1px solid #00aaff;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 10px;
    }
    /* Add padding around chat area */
    .stChat {
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown(
    "<h1>LegalClause: Contract Clause Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p>Enter a contract clause to classify it and assess risks using Legal-BERT and Gemini.</p>",
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages:
    for i, chat in enumerate(st.session_state.messages):
        message(chat['question'], is_user=True, key=f"user_{i}", avatar_style="identicon")
        message(chat['answer'], is_user=False, key=f"bot_{i}", avatar_style="micah")
else:
    st.markdown("No chat history yet. Start by entering a clause below.")

user_input = st.chat_input(placeholder="Enter a contract clause...")

if user_input:
    with st.spinner('Analyzing your clause...'):
        response = classify_and_analyze_clause(user_input)
    st.session_state.messages.append({"question": user_input, "answer": response})
    st.rerun()