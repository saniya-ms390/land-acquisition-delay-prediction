import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Land Acquisition Delay Prediction System",
    page_icon="🏗️",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 16px;
    color: #666;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SIDEBAR: DATA SOURCE & NAVIGATION
# ---------------------------------------------------

st.sidebar.title("🏗️ Navigation & Data")

data_source = st.sidebar.radio(
    "Select Data Source",
    ["Use Real-World Benchmark Dataset", "Upload Custom Survey CSV"]
)

df = None

if data_source == "Use Real-World Benchmark Dataset":
    data = {
        "compensation_pending": [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "legal_dispute":         [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        "documents_incomplete":  [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        "approval_pending":      [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "landowner_dispute":     [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        "delayed":               [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    st.sidebar.success("Loaded real benchmark survey dataset (25 records).")

else:
    uploaded_file = st.sidebar.file_uploader("Upload your survey CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom survey dataset successfully loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("Awaiting CSV upload. Using fallback benchmark dataset.")
        data = {
            "compensation_pending": [1, 0, 1, 1, 0],
            "legal_dispute": [1, 0, 1, 0, 0],
            "documents_incomplete": [1, 0, 1, 1, 0],
            "approval_pending": [1, 0, 1, 1, 0],
            "landowner_dispute": [1, 0, 1, 0, 0],
            "delayed": [1, 0, 1, 1, 0]
        }
        df = pd.DataFrame(data)

st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Predict Delay",
        "Risk Analysis",
        "Case Summary",
        "Model Information",
        "About Project"
    ]
)


# ---------------------------------------------------
# MACHINE LEARNING MODEL TRAINING
# ---------------------------------------------------

features = [
    "compensation_pending",
    "legal_dispute",
    "documents_incomplete",
    "approval_pending",
    "landowner_dispute"
]

X = df[features]
y = df["delayed"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

if page == "Dashboard":
    st.markdown('<div class="main-title">🏗️ Land Acquisition Delay Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Early risk detection powered by machine learning and real survey parameters.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cases / Records", len(df))

    with col2:
        st.metric("Total Delay Cases", int(df["delayed"].sum()))

    with col3:
        st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")

    st.divider()

    st.subheader("📌 Project Objective")
    st.write(
        """
        This system uses empirical risk factors derived from infrastructure surveys 
        to help planners flag potential bottlenecks and legal stalls prior to project execution.
        """
    )

    st.subheader("🔍 Key Risk Factors Analyzed")
    factor_data = pd.DataFrame({
        "Risk Factor": [
            "Compensation Pending",
            "Legal Dispute",
            "Incomplete Documents",
            "Approval Pending",
            "Landowner Dispute"
        ],
        "Description": [
            "Delays in fund disbursement or escrow account settlements.",
            "Active litigation or court stay orders against land parcels.",
            "Discrepancies in historical land titles and missing records.",
            "Statutory clearance delays from government bodies.",
            "Local agitations, protests, or resistance to land valuation."
        ]
    })
    st.dataframe(factor_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------
# PREDICT DELAY
# ---------------------------------------------------

elif page == "Predict Delay":
    st.title("🔮 Predict Land Acquisition Delay")
    st.write("Input current parameters from field survey reports:")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        compensation = st.selectbox("💰 Compensation Pending?", ["No", "Yes"])
        legal = st.selectbox("⚖️ Legal Dispute?", ["No", "Yes"])
        documents = st.selectbox("📄 Documents Incomplete?", ["No", "Yes"])

    with col2:
        approval = st.selectbox("📋 Approval Pending?", ["No", "Yes"])
        landowner = st.selectbox("👥 Landowner Dispute?", ["No", "Yes"])

    st.divider()

    if st.button("🔍 Predict Delay Risk", type="primary"):
        input_data = pd.DataFrame({
            "compensation_pending": [1 if compensation == "Yes" else 0],
            "legal_dispute": [1 if legal == "Yes" else 0],
            "documents_incomplete": [1 if documents == "Yes" else 0],
            "approval_pending": [1 if approval == "Yes" else 0],
            "landowner_dispute": [1 if landowner == "Yes" else 0]
        })

        probability = model.predict_proba(input_data)[0][1]

        if probability >= 0.70:
            st.error("🚨 HIGH RISK: High probability of significant project delay.")
        elif probability >= 0.40:
            st.warning("⚠️ MEDIUM RISK: Moderate delay indicators present. Monitor closely.")
        else:
            st.success("✅ LOW RISK: Land parcel is clear for timely acquisition.")

        st.metric("Estimated Delay Probability", f"{probability * 100:.1f}%")

        st.subheader("💡 Recommended Actions")
        if compensation == "Yes":
            st.write("• Fast-track compensation disbursements and resolve escrow discrepancies.")
        if legal == "Yes":
            st.write("• Engage legal advisors to push for fast-track court clearance or settlements.")
        if documents == "Yes":
            st.write("• Conduct intensive title verification and digitize missing records.")
        if approval == "Yes":
            st.write("• Coordinate with nodal statutory agencies for pending sign-offs.")
        if landowner == "Yes":
            st.write("• Setup direct negotiation panels and community engagement forums.")
        if compensation == "No" and legal == "No" and documents == "No" and approval == "No" and landowner == "No":
            st.write("• Maintain routine milestone tracking.")


# ---------------------------------------------------
# RISK ANALYSIS
# ---------------------------------------------------

elif page == "Risk Analysis":
    st.title("📊 Feature & Risk Factor Analysis")
    st.write("Relative importance of input features derived from the trained Random Forest model:")

    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "Risk Factor": [
            "Compensation Pending",
            "Legal Dispute",
            "Documents Incomplete",
            "Approval Pending",
            "Landowner Dispute"
        ],
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Risk Factor"))

    display_df = importance_df.copy()
    display_df["Importance (%)"] = (display_df["Importance"] * 100).round(2)
    display_df = display_df.drop(columns=["Importance"])

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------
# CASE SUMMARY
# ---------------------------------------------------

elif page == "Case Summary":
    st.title("📋 Survey Dataset Records")
    st.write("Detailed view of survey observations loaded into the evaluation pipeline:")

    summary_df = df.copy()
    summary_df["Delayed"] = summary_df["delayed"].map({0: "No", 1: "Yes"})
    summary_df = summary_df.drop(columns=["delayed"])

    for col in summary_df.columns:
        if col != "Delayed":
            summary_df[col] = summary_df[col].map({0: "No", 1: "Yes"})

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current Dataset as CSV",
        data=csv_data,
        file_name="land_acquisition_survey_data.csv",
        mime="text/csv"
    )


# ---------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------

elif page == "Model Information":
    st.title("🤖 Machine Learning Model Performance")
    st.subheader("Algorithm: Random Forest Classifier")
    st.write("Ensemble decision tree framework built to handle non-linear survey attributes effectively.")

    st.metric("Test Set Accuracy", f"{accuracy * 100:.1f}%")

    st.subheader("Detailed Classification Report")
    try:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    except Exception:
        st.info("Insufficient test sample volume for full classification breakdown.")


# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

elif page == "About Project":
    st.title("ℹ️ About the Project")
    st.write(
        """
        ### AI/ML Based Land Acquisition Delay Prediction System
        Designed to assist public works authorities and project managers by anticipating bottlenecks 
        caused by delayed compensation payouts, documentation errors, and title litigations.
        """
    )
    st.subheader("🛠️ Tech Stack")
    for tech in ["Python", "Streamlit", "Pandas", "Scikit-Learn", "Random Forest"]:
        st.write(f"• {tech}")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()
st.caption("Land Acquisition Delay Prediction System | AIML Project")