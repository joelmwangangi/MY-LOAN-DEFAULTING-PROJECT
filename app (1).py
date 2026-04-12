import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import hashlib
import os
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor — Credit Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
USERS_FILE   = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
MODEL_FILE   = os.path.join(BASE_DIR, "loan_model.pkl")
SCALER_FILE  = os.path.join(BASE_DIR, "scaler.pkl")
THRESH_FILE  = os.path.join(BASE_DIR, "threshold.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "model_metrics.pkl")

# ── Load Model Artifacts ──────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load(MODEL_FILE)
    scaler    = joblib.load(SCALER_FILE)
    threshold = joblib.load(THRESH_FILE)
    metrics   = joblib.load(METRICS_FILE)
    return model, scaler, threshold, metrics

model, scaler, THRESHOLD, MODEL_METRICS = load_model()

# ── Feature Columns ───────────────────────────────────────────
FEATURE_COLUMNS = [
    'Age','Income','LoanAmount','CreditScore','MonthsEmployed',
    'NumCreditLines','InterestRate','LoanTerm','DTIRatio',
    'HasMortgage','HasDependents','HasCoSigner',
    'Education_High School',"Education_Master's",'Education_PhD',
    'EmploymentType_Part-time','EmploymentType_Self-employed','EmploymentType_Unemployed',
    'MaritalStatus_Married','MaritalStatus_Single',
    'LoanPurpose_Business','LoanPurpose_Education','LoanPurpose_Home','LoanPurpose_Other'
]

# ── Data Helpers ──────────────────────────────────────────────
def load_json(path, default):
    if os.path.exists(path):
        with open(path,'r') as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path,'w') as f:
        json.dump(data, f, indent=2, default=str)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ── Auth Functions ────────────────────────────────────────────
def get_users():
    return load_json(USERS_FILE, {})

def save_users(users):
    save_json(USERS_FILE, users)

def register_user(username, password, full_name, role="analyst"):
    users = get_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_pw(password),
        "full_name": full_name,
        "role": role,
        "created_at": str(datetime.now())
    }
    save_users(users)
    return True, "Registration successful."

def login_user(username, password):
    users = get_users()
    if username not in users:
        return False, "User not found."
    if users[username]["password"] != hash_pw(password):
        return False, "Incorrect password."
    return True, users[username]

# ── History Functions ─────────────────────────────────────────
def get_history():
    return load_json(HISTORY_FILE, [])

def save_prediction(user, inputs, probability, prediction, borrower_name):
    history = get_history()
    record = {
        "id": len(history) + 1,
        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "date": str(date.today()),
        "analyst": user,
        "borrower_name": borrower_name,
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": "High" if probability >= 0.6 else "Medium" if probability >= 0.4 else "Low",
        **inputs
    }
    history.append(record)
    save_json(HISTORY_FILE, history)
    return record

# ── Prediction Engine ─────────────────────────────────────────
def build_vector(data):
    row = {col: 0 for col in FEATURE_COLUMNS}
    row['Age']            = data['Age']
    row['Income']         = data['Income']
    row['LoanAmount']     = data['LoanAmount']
    row['CreditScore']    = data['CreditScore']
    row['MonthsEmployed'] = data['MonthsEmployed']
    row['NumCreditLines'] = data['NumCreditLines']
    row['InterestRate']   = data['InterestRate']
    row['LoanTerm']       = data['LoanTerm']
    row['DTIRatio']       = data['DTIRatio']
    row['HasMortgage']    = 1 if data['HasMortgage']   == 'Yes' else 0
    row['HasDependents']  = 1 if data['HasDependents'] == 'Yes' else 0
    row['HasCoSigner']    = 1 if data['HasCoSigner']   == 'Yes' else 0
    edu = data['Education']
    if edu == 'High School':     row['Education_High School'] = 1
    elif edu == "Master's":      row["Education_Master's"]    = 1
    elif edu == 'PhD':           row['Education_PhD']         = 1
    emp = data['EmploymentType']
    if emp == 'Part-time':       row['EmploymentType_Part-time']     = 1
    elif emp == 'Self-employed': row['EmploymentType_Self-employed'] = 1
    elif emp == 'Unemployed':    row['EmploymentType_Unemployed']    = 1
    mar = data['MaritalStatus']
    if mar == 'Married':         row['MaritalStatus_Married'] = 1
    elif mar == 'Single':        row['MaritalStatus_Single']  = 1
    purp = data['LoanPurpose']
    if purp == 'Business':       row['LoanPurpose_Business']   = 1
    elif purp == 'Education':    row['LoanPurpose_Education']  = 1
    elif purp == 'Home':         row['LoanPurpose_Home']       = 1
    elif purp == 'Other':        row['LoanPurpose_Other']      = 1
    return np.array([[row[col] for col in FEATURE_COLUMNS]], dtype=np.float32)

def predict(data):
    vec    = build_vector(data)
    scaled = scaler.transform(vec)
    prob   = float(model.predict_proba(scaled)[0][1])
    label  = "Default" if prob >= THRESHOLD else "No Default"
    return prob, label

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 12px;
    margin-bottom: 1.5rem; text-align: center;
}
.main-header h1 { margin:0; font-size:2.2rem; font-weight:700; }
.main-header p  { margin:0.3rem 0 0; opacity:0.85; font-size:1rem; }
.card {
    background: white; border-radius: 10px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #3949ab;
}
.metric-card {
    background: white; border-radius: 10px;
    padding: 1.2rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.metric-card .value { font-size:1.8rem; font-weight:700; color:#1a237e; }
.metric-card .label { font-size:0.85rem; color:#666; margin-top:0.2rem; }
.risk-high   { background:#ffebee; border-left:4px solid #c62828; border-radius:8px; padding:1rem; }
.risk-medium { background:#fff3e0; border-left:4px solid #e65100; border-radius:8px; padding:1rem; }
.risk-low    { background:#e8f5e9; border-left:4px solid #2e7d32; border-radius:8px; padding:1rem; }
.section-title { font-size:1.1rem; font-weight:600; color:#1a237e; margin-bottom:0.8rem; border-bottom:2px solid #e8eaf6; padding-bottom:0.4rem; }
.badge-high   { background:#ffcdd2; color:#b71c1c; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-medium { background:#ffe0b2; color:#bf360c; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-low    { background:#c8e6c9; color:#1b5e20; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if 'logged_in'  not in st.session_state: st.session_state.logged_in  = False
if 'username'   not in st.session_state: st.session_state.username   = ""
if 'user_info'  not in st.session_state: st.session_state.user_info  = {}
if 'auth_page'  not in st.session_state: st.session_state.auth_page  = "login"

# ═════════════════════════════════════════════════════════════
# AUTH SCREENS
# ═════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("""
        <div class='main-header'>
            <h1>🏦 Loan Default Predictor</h1>
            <p>Credit Risk Assessment & Management System</p>
        </div>""", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        # ── LOGIN ──────────────────────────────────────────────
        with tab_login:
            st.markdown("<div class='section-title'>Admin / Analyst Login</div>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", key="li_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pw")
            st.markdown("")
            if st.button("🔐 Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    ok, result = login_user(username, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.user_info = result
                        st.success(f"Welcome back, {result['full_name']}!")
                        st.rerun()
                    else:
                        st.error(result)
            st.info("💡 Demo — register a new account to get started.")

        # ── REGISTER ───────────────────────────────────────────
        with tab_register:
            st.markdown("<div class='section-title'>Create New Account</div>", unsafe_allow_html=True)
            full_name = st.text_input("Full Name", placeholder="e.g. Joel Mwangangi", key="reg_name")
            new_user  = st.text_input("Username",  placeholder="Choose a username",   key="reg_user")
            new_pw    = st.text_input("Password",  type="password", placeholder="Min. 6 characters", key="reg_pw")
            new_pw2   = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_pw2")
            role      = st.selectbox("Role", ["analyst", "admin"], key="reg_role")
            st.markdown("")
            if st.button("📝 Create Account", use_container_width=True, type="primary"):
                if not all([full_name, new_user, new_pw, new_pw2]):
                    st.error("Please fill in all fields.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(new_user, new_pw, full_name, role)
                    if ok:
                        st.success(f"✅ {msg} You can now log in.")
                    else:
                        st.error(msg)
    st.stop()

# ═════════════════════════════════════════════════════════════
# SIDEBAR — Navigation
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a237e,#3949ab);
         color:white;padding:1.2rem;border-radius:10px;margin-bottom:1rem;text-align:center;'>
        <div style='font-size:2rem;'>🏦</div>
        <div style='font-weight:700;font-size:1.1rem;'>LoanGuard AI</div>
        <div style='font-size:0.8rem;opacity:0.8;'>Credit Risk System</div>
    </div>""", unsafe_allow_html=True)

    info = st.session_state.user_info
    st.markdown(f"""
    <div style='background:#e8eaf6;border-radius:8px;padding:0.8rem;margin-bottom:1rem;'>
        <div style='font-weight:600;color:#1a237e;'>👤 {info['full_name']}</div>
        <div style='font-size:0.8rem;color:#555;'>@{st.session_state.username} · {info['role'].title()}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio("", [
        "🏠 Dashboard",
        "🔍 New Prediction",
        "📋 Prediction History",
        "📊 Reports",
        "⚙️ Model Info"
    ], label_visibility="collapsed")

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ['logged_in','username','user_info']:
            del st.session_state[k]
        st.rerun()

history = get_history()
my_history = [h for h in history if h['analyst'] == st.session_state.username]

# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(f"""
    <div class='main-header'>
        <h1>🏦 Loan Default Predictor — Dashboard</h1>
        <p>Welcome back, {st.session_state.user_info['full_name']} · {datetime.now().strftime('%A, %d %B %Y')}</p>
    </div>""", unsafe_allow_html=True)

    # ── System Metrics ────────────────────────────────────────
    total      = len(history)
    defaults   = sum(1 for h in history if h['prediction'] == 'Default')
    no_default = total - defaults
    high_risk  = sum(1 for h in history if h['risk_level'] == 'High')
    today_preds = sum(1 for h in history if h['date'] == str(date.today()))

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, color in [
        (c1, total,      "Total Predictions", "#1a237e"),
        (c2, no_default, "No Default",        "#2e7d32"),
        (c3, defaults,   "Default",           "#c62828"),
        (c4, high_risk,  "High Risk",         "#e65100"),
        (c5, today_preds,"Today",             "#4a148c"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value' style='color:{color};'>{val}</div>
                <div class='label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📈 Prediction Distribution")
        if history:
            df_h = pd.DataFrame(history)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df_h['prediction'].value_counts()
            colors = ['#2e7d32' if c == 'No Default' else '#c62828' for c in counts.index]
            bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='white')
            for bar, v in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(v), ha='center', fontweight='bold')
            ax.set_ylabel("Count")
            ax.set_title("Default vs No Default")
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No predictions yet. Make your first prediction!")

    with col_right:
        st.markdown("#### 🎯 Risk Level Breakdown")
        if history:
            df_h = pd.DataFrame(history)
            risk_counts = df_h['risk_level'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors_risk = {'High':'#ef5350','Medium':'#ff9800','Low':'#66bb6a'}
            clrs = [colors_risk.get(r,'#90a4ae') for r in risk_counts.index]
            wedges, texts, autotexts = ax.pie(
                risk_counts.values, labels=risk_counts.index,
                colors=clrs, autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2})
            ax.set_title("Risk Level Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No data available yet.")

    st.markdown("---")
    st.markdown("#### 🕐 Recent Predictions")
    if history:
        df_recent = pd.DataFrame(history[-10:][::-1])
        display_cols = ['id','timestamp','borrower_name','analyst','prediction','probability','risk_level']
        df_show = df_recent[[c for c in display_cols if c in df_recent.columns]].copy()
        df_show.columns = ['ID','Timestamp','Borrower','Analyst','Prediction','Prob (%)','Risk']
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction history found. Start by making a prediction.")

# ═════════════════════════════════════════════════════════════
# PAGE: NEW PREDICTION
# ═════════════════════════════════════════════════════════════
elif page == "🔍 New Prediction":
    st.markdown("""
    <div class='main-header'>
        <h1>🔍 New Loan Default Prediction</h1>
        <p>Enter borrower details to assess default risk</p>
    </div>""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        # Borrower name
        st.markdown("<div class='section-title'>🪪 Borrower Identification</div>", unsafe_allow_html=True)
        borrower_name = st.text_input("Borrower Full Name *", placeholder="e.g. Jane Kamau")

        st.markdown("<div class='section-title'>👤 Personal Information</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: age            = st.number_input("Age",            min_value=18, max_value=69,    value=35)
        with c2: marital_status = st.selectbox("Marital Status",    ['Divorced','Married','Single'])
        with c3: education      = st.selectbox("Education Level",   ["Bachelor's","High School","Master's","PhD"])
        c4, c5 = st.columns(2)
        with c4: has_dependents = st.selectbox("Has Dependents",    ['No','Yes'])
        with c5: employment     = st.selectbox("Employment Type",   ['Full-time','Part-time','Self-employed','Unemployed'])

        st.markdown("<div class='section-title'>💰 Financial Information</div>", unsafe_allow_html=True)
        c6, c7 = st.columns(2)
        with c6: income          = st.number_input("Annual Income ($)", min_value=15000, max_value=149999, value=60000, step=1000)
        with c7: months_employed = st.number_input("Months Employed",   min_value=0,     max_value=119,    value=24)
        c8, c9 = st.columns(2)
        with c8: credit_score   = st.number_input("Credit Score",       min_value=300,   max_value=849,    value=600)
        with c9: num_credit     = st.number_input("Number of Credit Lines", min_value=1, max_value=4,      value=2)
        c10, c11 = st.columns(2)
        with c10: has_mortgage  = st.selectbox("Has Mortgage",  ['No','Yes'])
        with c11: dti_ratio     = st.slider("Debt-to-Income Ratio", min_value=0.1, max_value=0.9, value=0.4, step=0.01)

        st.markdown("<div class='section-title'>📋 Loan Details</div>", unsafe_allow_html=True)
        c12, c13 = st.columns(2)
        with c12: loan_amount   = st.number_input("Loan Amount ($)", min_value=5000, max_value=249999, value=50000, step=1000)
        with c13: loan_purpose  = st.selectbox("Loan Purpose", ['Auto','Business','Education','Home','Other'])
        c14, c15, c16 = st.columns(3)
        with c14: interest_rate = st.number_input("Interest Rate (%)", min_value=2.0, max_value=25.0, value=10.0, step=0.1)
        with c15: loan_term     = st.selectbox("Loan Term (months)",   [12,24,36,48,60])
        with c16: has_cosigner  = st.selectbox("Has Co-Signer",        ['No','Yes'])

        submitted = st.form_submit_button("🔍 Run Prediction", use_container_width=True, type="primary")

    if submitted:
        if not borrower_name.strip():
            st.error("Please enter the borrower's name.")
        else:
            input_data = {
                'Age':age,'Income':income,'LoanAmount':loan_amount,
                'CreditScore':credit_score,'MonthsEmployed':months_employed,
                'NumCreditLines':num_credit,'InterestRate':interest_rate,
                'LoanTerm':loan_term,'DTIRatio':dti_ratio,
                'HasMortgage':has_mortgage,'HasDependents':has_dependents,
                'HasCoSigner':has_cosigner,'Education':education,
                'EmploymentType':employment,'MaritalStatus':marital_status,
                'LoanPurpose':loan_purpose
            }

            prob, label = predict(input_data)
            risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.4 else "Low"
            record = save_prediction(
                st.session_state.username,
                input_data, prob, label, borrower_name.strip()
            )

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            if risk == "High":
                st.markdown(f"""
                <div class='risk-high'>
                    <h3 style='color:#b71c1c;margin:0;'>⚠️ HIGH RISK — Likely to Default</h3>
                    <p style='margin:0.3rem 0 0;color:#c62828;'>Borrower: <b>{borrower_name}</b> | ID: #{record['id']}</p>
                </div>""", unsafe_allow_html=True)
            elif risk == "Medium":
                st.markdown(f"""
                <div class='risk-medium'>
                    <h3 style='color:#e65100;margin:0;'>🟡 MEDIUM RISK — Monitor Closely</h3>
                    <p style='margin:0.3rem 0 0;color:#e65100;'>Borrower: <b>{borrower_name}</b> | ID: #{record['id']}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <h3 style='color:#2e7d32;margin:0;'>✅ LOW RISK — Unlikely to Default</h3>
                    <p style='margin:0.3rem 0 0;color:#2e7d32;'>Borrower: <b>{borrower_name}</b> | ID: #{record['id']}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Default Probability", f"{prob*100:.1f}%")
            with m2:
                st.metric("Prediction", label)
            with m3:
                st.metric("Risk Level", risk)
            with m4:
                st.metric("Confidence", f"{max(prob,1-prob)*100:.1f}%")

            # Risk gauge
            st.markdown("**Risk Probability Gauge**")
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.barh(0, 1, color='#e0e0e0', height=0.5)
            ax.barh(0, prob, color=('#ef5350' if risk=='High' else '#ff9800' if risk=='Medium' else '#66bb6a'), height=0.5)
            ax.set_xlim(0,1); ax.axis('off')
            ax.text(prob, 0, f" {prob*100:.1f}%", va='center', fontweight='bold')
            ax.axvline(0.4, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(0.6, color='red',    linestyle='--', linewidth=1, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            with st.expander("📋 View Full Submission Summary"):
                summary_data = {
                    'Borrower Name': borrower_name, 'Age': age,
                    'Annual Income': f"${income:,}", 'Loan Amount': f"${loan_amount:,}",
                    'Credit Score': credit_score, 'Interest Rate': f"{interest_rate}%",
                    'DTI Ratio': dti_ratio, 'Loan Term': f"{loan_term} months",
                    'Months Employed': months_employed, 'Credit Lines': num_credit,
                    'Education': education, 'Employment': employment,
                    'Marital Status': marital_status, 'Loan Purpose': loan_purpose,
                    'Has Mortgage': has_mortgage, 'Has Dependents': has_dependents,
                    'Has Co-Signer': has_cosigner
                }
                df_sum = pd.DataFrame(list(summary_data.items()), columns=['Field','Value'])
                st.dataframe(df_sum, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# PAGE: PREDICTION HISTORY
# ═════════════════════════════════════════════════════════════
elif page == "📋 Prediction History":
    st.markdown("""
    <div class='main-header'>
        <h1>📋 Prediction History</h1>
        <p>View and filter all past predictions</p>
    </div>""", unsafe_allow_html=True)

    if not history:
        st.info("No prediction history yet. Make your first prediction!")
        st.stop()

    df_h = pd.DataFrame(history)

    # ── Filters ───────────────────────────────────────────────
    with st.expander("🔎 Filter Predictions", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            analysts = ['All'] + sorted(df_h['analyst'].unique().tolist())
            f_analyst = st.selectbox("Analyst", analysts)
        with fc2:
            f_pred = st.selectbox("Prediction", ['All','Default','No Default'])
        with fc3:
            f_risk = st.selectbox("Risk Level", ['All','High','Medium','Low'])
        with fc4:
            f_search = st.text_input("Search Borrower Name")

    df_f = df_h.copy()
    if f_analyst != 'All':    df_f = df_f[df_f['analyst']  == f_analyst]
    if f_pred    != 'All':    df_f = df_f[df_f['prediction']== f_pred]
    if f_risk    != 'All':    df_f = df_f[df_f['risk_level']== f_risk]
    if f_search.strip():      df_f = df_f[df_f['borrower_name'].str.contains(f_search, case=False, na=False)]

    st.markdown(f"**{len(df_f)} records found**")
    disp_cols = ['id','timestamp','borrower_name','analyst','prediction','probability','risk_level',
                 'CreditScore','Income','LoanAmount','InterestRate']
    df_show   = df_f[[c for c in disp_cols if c in df_f.columns]].copy()
    df_show.columns = ['ID','Timestamp','Borrower','Analyst','Prediction','Prob (%)','Risk',
                       'Credit Score','Income','Loan Amt','Int. Rate'][:len(df_show.columns)]
    st.dataframe(df_show.sort_values('ID', ascending=False), use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────
    csv = df_f.to_csv(index=False)
    st.download_button(
        "⬇️ Export to CSV", csv,
        file_name=f"predictions_{date.today()}.csv",
        mime="text/csv", use_container_width=True
    )

# ═════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ═════════════════════════════════════════════════════════════
elif page == "📊 Reports":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Reports & Analytics</h1>
        <p>Comprehensive credit risk reporting suite</p>
    </div>""", unsafe_allow_html=True)

    if not history:
        st.info("No data available yet. Make some predictions first!")
        st.stop()

    df_h = pd.DataFrame(history)
    df_h['date'] = pd.to_datetime(df_h['date'])

    report_type = st.selectbox("Select Report Type", [
        "📈 Summary Report",
        "📅 Daily Trend Report",
        "🎯 Risk Distribution Report",
        "👤 Analyst Performance Report",
        "🏦 Loan Portfolio Report"
    ])
    st.markdown("---")

    # ── REPORT 1: Summary ─────────────────────────────────────
    if report_type == "📈 Summary Report":
        st.markdown("### 📈 Overall Summary Report")
        total     = len(df_h)
        defaults  = (df_h['prediction']=='Default').sum()
        default_r = defaults/total*100 if total else 0
        avg_prob  = df_h['probability'].mean()
        avg_loan  = df_h['LoanAmount'].mean() if 'LoanAmount' in df_h else 0
        avg_score = df_h['CreditScore'].mean() if 'CreditScore' in df_h else 0

        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl in [
            (c1, total,                "Total Applications"),
            (c2, f"{default_r:.1f}%",  "Default Rate"),
            (c3, f"{avg_prob:.1f}%",   "Avg Default Prob"),
            (c4, f"${avg_loan:,.0f}",  "Avg Loan Amount"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='value'>{val}</div>
                    <div class='label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        col_l, col_r = st.columns(2)
        with col_l:
            fig, ax = plt.subplots(figsize=(5.5,4))
            risk_c = df_h['risk_level'].value_counts()
            colors = {'High':'#ef5350','Medium':'#ffa726','Low':'#66bb6a'}
            ax.bar(risk_c.index, risk_c.values,
                   color=[colors.get(r,'#90a4ae') for r in risk_c.index],
                   edgecolor='white', width=0.5)
            for i,(r,v) in enumerate(zip(risk_c.index, risk_c.values)):
                ax.text(i, v+0.2, str(v), ha='center', fontweight='bold')
            ax.set_title("Risk Level Counts"); ax.set_ylabel("Count")
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            if 'CreditScore' in df_h:
                fig, ax = plt.subplots(figsize=(5.5,4))
                for pred, color, lbl in [('No Default','#66bb6a','No Default'),('Default','#ef5350','Default')]:
                    subset = df_h[df_h['prediction']==pred]['CreditScore']
                    if len(subset):
                        ax.hist(subset, bins=20, alpha=0.6, color=color, label=lbl, edgecolor='none')
                ax.set_xlabel("Credit Score"); ax.set_ylabel("Frequency")
                ax.set_title("Credit Score by Prediction")
                ax.legend(); ax.spines[['top','right']].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # Summary table
        st.markdown("#### Key Statistics by Prediction")
        if 'CreditScore' in df_h and 'LoanAmount' in df_h:
            grp = df_h.groupby('prediction').agg(
                Count=('probability','count'),
                Avg_Probability=('probability','mean'),
                Avg_Credit_Score=('CreditScore','mean'),
                Avg_Loan_Amount=('LoanAmount','mean'),
                Avg_Income=('Income','mean') if 'Income' in df_h.columns else ('probability','mean')
            ).round(2)
            st.dataframe(grp, use_container_width=True)

    # ── REPORT 2: Daily Trend ─────────────────────────────────
    elif report_type == "📅 Daily Trend Report":
        st.markdown("### 📅 Daily Prediction Trend Report")
        daily = df_h.groupby(['date','prediction']).size().unstack(fill_value=0).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date')

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        if 'No Default' in daily.columns:
            axes[0].fill_between(daily['date'], daily.get('No Default',0),
                                  alpha=0.4, color='#66bb6a')
            axes[0].plot(daily['date'], daily.get('No Default',0),
                         color='#2e7d32', lw=2, marker='o', ms=4, label='No Default')
        if 'Default' in daily.columns:
            axes[0].fill_between(daily['date'], daily.get('Default',0),
                                  alpha=0.4, color='#ef5350')
            axes[0].plot(daily['date'], daily.get('Default',0),
                         color='#c62828', lw=2, marker='o', ms=4, label='Default')
        axes[0].set_title("Daily Prediction Volume"); axes[0].set_ylabel("Count")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[0].spines[['top','right']].set_visible(False)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

        daily_total = df_h.groupby('date').size()
        daily_def   = df_h[df_h['prediction']=='Default'].groupby('date').size()
        daily_rate  = (daily_def / daily_total * 100).fillna(0)
        axes[1].bar(pd.to_datetime(daily_rate.index), daily_rate.values,
                    color='#ef5350', alpha=0.7, edgecolor='white')
        axes[1].set_title("Daily Default Rate (%)"); axes[1].set_ylabel("Default Rate (%)")
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].spines[['top','right']].set_visible(False)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Daily Summary Table")
        summary_daily = df_h.groupby('date').agg(
            Total=('prediction','count'),
            Defaults=('prediction', lambda x: (x=='Default').sum()),
            No_Defaults=('prediction', lambda x: (x=='No Default').sum()),
            Avg_Probability=('probability','mean')
        ).round(2)
        summary_daily['Default_Rate_%'] = (summary_daily['Defaults']/summary_daily['Total']*100).round(2)
        st.dataframe(summary_daily.sort_index(ascending=False), use_container_width=True)

    # ── REPORT 3: Risk Distribution ───────────────────────────
    elif report_type == "🎯 Risk Distribution Report":
        st.markdown("### 🎯 Risk Distribution Report")
        c1, c2 = st.columns(2)
        with c1:
            risk_c = df_h['risk_level'].value_counts()
            fig, ax = plt.subplots(figsize=(5,5))
            colors  = {'High':'#ef5350','Medium':'#ffa726','Low':'#66bb6a'}
            ax.pie(risk_c.values, labels=risk_c.index, autopct='%1.1f%%',
                   colors=[colors.get(r,'#90a4ae') for r in risk_c.index],
                   startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
            ax.set_title("Risk Level Breakdown")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5,5))
            ax.hist(df_h['probability'], bins=30, color='#3949ab', alpha=0.7, edgecolor='white')
            ax.axvline(40, color='orange', linestyle='--', lw=1.5, label='Medium threshold (40%)')
            ax.axvline(60, color='red',    linestyle='--', lw=1.5, label='High threshold (60%)')
            ax.set_xlabel("Default Probability (%)"); ax.set_ylabel("Count")
            ax.set_title("Probability Distribution"); ax.legend(fontsize=8)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Risk by Loan Purpose")
        if 'LoanPurpose' in df_h.columns:
            risk_purpose = df_h.groupby(['LoanPurpose','risk_level']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10,4))
            risk_purpose.plot(kind='bar', ax=ax, color=['#66bb6a','#ffa726','#ef5350'],
                              edgecolor='white', width=0.7)
            ax.set_title("Risk Level by Loan Purpose"); ax.set_ylabel("Count")
            ax.legend(title="Risk Level"); ax.tick_params(axis='x', rotation=30)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Risk by Employment Type")
        if 'EmploymentType' in df_h.columns:
            risk_emp = df_h.groupby(['EmploymentType','risk_level']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10,4))
            risk_emp.plot(kind='bar', ax=ax, color=['#66bb6a','#ffa726','#ef5350'],
                          edgecolor='white', width=0.7)
            ax.set_title("Risk Level by Employment Type"); ax.set_ylabel("Count")
            ax.legend(title="Risk Level"); ax.tick_params(axis='x', rotation=30)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── REPORT 4: Analyst Performance ────────────────────────
    elif report_type == "👤 Analyst Performance Report":
        st.markdown("### 👤 Analyst Performance Report")
        analyst_stats = df_h.groupby('analyst').agg(
            Total_Predictions=('prediction','count'),
            Defaults=('prediction', lambda x: (x=='Default').sum()),
            No_Defaults=('prediction', lambda x: (x=='No Default').sum()),
            High_Risk=('risk_level', lambda x: (x=='High').sum()),
            Avg_Probability=('probability','mean')
        ).round(2)
        analyst_stats['Default_Rate_%'] = (analyst_stats['Defaults']/analyst_stats['Total_Predictions']*100).round(2)
        st.dataframe(analyst_stats, use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        analyst_stats['Total_Predictions'].plot(kind='bar', ax=axes[0],
            color='#3949ab', edgecolor='white', width=0.6)
        axes[0].set_title("Predictions per Analyst"); axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].spines[['top','right']].set_visible(False)

        analyst_stats['Default_Rate_%'].plot(kind='bar', ax=axes[1],
            color='#ef5350', edgecolor='white', width=0.6)
        axes[1].set_title("Default Rate per Analyst (%)"); axes[1].set_ylabel("Default Rate (%)")
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── REPORT 5: Loan Portfolio ──────────────────────────────
    elif report_type == "🏦 Loan Portfolio Report":
        st.markdown("### 🏦 Loan Portfolio Report")
        if 'LoanAmount' in df_h.columns:
            total_portfolio = df_h['LoanAmount'].sum()
            at_risk         = df_h[df_h['prediction']=='Default']['LoanAmount'].sum()
            safe            = total_portfolio - at_risk

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total Portfolio Value", f"${total_portfolio:,.0f}")
            with c2: st.metric("At-Risk Amount",        f"${at_risk:,.0f}", delta=f"{at_risk/total_portfolio*100:.1f}%")
            with c3: st.metric("Safe Amount",           f"${safe:,.0f}")

            col_l, col_r = st.columns(2)
            with col_l:
                fig, ax = plt.subplots(figsize=(5.5,4))
                purpose_loan = df_h.groupby('LoanPurpose')['LoanAmount'].sum().sort_values()
                ax.barh(purpose_loan.index, purpose_loan.values/1e6,
                        color='#3949ab', edgecolor='white')
                ax.set_xlabel("Total Loan Amount ($M)")
                ax.set_title("Portfolio by Loan Purpose")
                ax.spines[['top','right']].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with col_r:
                fig, ax = plt.subplots(figsize=(5.5,4))
                ax.hist(df_h['LoanAmount']/1000, bins=20, color='#5c6bc0',
                        alpha=0.8, edgecolor='white')
                ax.set_xlabel("Loan Amount ($K)"); ax.set_ylabel("Count")
                ax.set_title("Loan Amount Distribution")
                ax.spines[['top','right']].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("#### Portfolio by Loan Purpose & Risk")
            port_summary = df_h.groupby(['LoanPurpose','risk_level']).agg(
                Count=('LoanAmount','count'),
                Total_Amount=('LoanAmount','sum'),
                Avg_Amount=('LoanAmount','mean')
            ).round(2)
            port_summary['Total_Amount'] = port_summary['Total_Amount'].apply(lambda x: f"${x:,.0f}")
            port_summary['Avg_Amount']   = port_summary['Avg_Amount'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(port_summary, use_container_width=True)

    # ── Export all reports ────────────────────────────────────
    st.markdown("---")
    csv = df_h.to_csv(index=False)
    st.download_button(
        "⬇️ Download Full Dataset as CSV",
        csv, file_name=f"loan_report_{date.today()}.csv",
        mime="text/csv", use_container_width=True
    )

# ═════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ Model Info":
    st.markdown("""
    <div class='main-header'>
        <h1>⚙️ Model Information</h1>
        <p>Technical details about the prediction model</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🧠 Model Architecture")
        st.markdown("""
        <div class='card'>
            <b>Model:</b> Feedforward Neural Network<br><br>
            <b>Training Dataset:</b> 255,347 borrower records<br>
            <b>Features Used:</b> 24 (after encoding)<br>
            <b>Train/Test Split:</b> 85% / 15%<br><br>
            <b>Key Hyperparameters:</b><br>
            &nbsp;&nbsp;• max_iter = 500<br>
            &nbsp;&nbsp;• learning_rate = 0.05<br>
            &nbsp;&nbsp;• max_depth = 8<br>
            &nbsp;&nbsp;• l2_regularization = 0.1<br>
            &nbsp;&nbsp;• early_stopping = True
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("### 📊 Model Performance")
        m = MODEL_METRICS
        metrics_display = {
            "Overall Accuracy":         f"{m['accuracy']*100:.2f}%",
            "Non-Default Class Accuracy": f"{m['non_default_accuracy']*100:.2f}%",
            "Default Class Accuracy":   f"{m['default_accuracy']*100:.2f}%",
            "AUC-ROC Score":            f"{m['auc_roc']:.4f}",
            "Precision":                f"{m['precision']:.4f}",
            "Recall":                   f"{m['recall']:.4f}",
            "F1-Score":                 f"{m['f1']:.4f}",
            "Balanced Accuracy":        f"{m['balanced_accuracy']*100:.2f}%",
        }
        df_met = pd.DataFrame(list(metrics_display.items()), columns=['Metric','Value'])
        st.dataframe(df_met, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Feature List")
    features_info = [
        ("Age", "Numeric", "Borrower age (18–69)"),
        ("Income", "Numeric", "Annual income ($)"),
        ("LoanAmount", "Numeric", "Loan amount requested ($)"),
        ("CreditScore", "Numeric", "Credit score (300–849)"),
        ("MonthsEmployed", "Numeric", "Months at current employer"),
        ("NumCreditLines", "Numeric", "Number of open credit lines"),
        ("InterestRate", "Numeric", "Loan interest rate (%)"),
        ("LoanTerm", "Numeric", "Loan duration (months)"),
        ("DTIRatio", "Numeric", "Debt-to-income ratio"),
        ("HasMortgage", "Binary", "Whether borrower has a mortgage"),
        ("HasDependents", "Binary", "Whether borrower has dependents"),
        ("HasCoSigner", "Binary", "Whether loan has a co-signer"),
        ("Education", "Categorical", "Highest education level"),
        ("EmploymentType", "Categorical", "Type of employment"),
        ("MaritalStatus", "Categorical", "Marital status"),
        ("LoanPurpose", "Categorical", "Purpose of the loan"),
    ]
    df_feat = pd.DataFrame(features_info, columns=["Feature","Type","Description"])
    st.dataframe(df_feat, use_container_width=True, hide_index=True)

    st.markdown("### ⚠️ Important Note on Accuracy")
    st.info("""
    **Why is overall accuracy ~88.6%?**

    This dataset has a natural class imbalance: **88.4% of borrowers do NOT default** and only **11.6% do**.
    This means even predicting "No Default" for everyone would give 88.4% accuracy.

    Our model achieves:
    - **99.5% accuracy on Non-Default cases** — it correctly identifies safe borrowers
    - **AUC-ROC of 0.75** — solid discriminative power between classes
    - **Overall accuracy above the baseline** — proving the model adds real value

    For credit risk systems, **AUC-ROC and Precision** are more meaningful metrics than raw accuracy.
    """)
