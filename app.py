
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# 2. TRAINING DATA
# Synthetic demonstration examples
# ==========================================

messages = [
    # Normal messages
    ("Tomorrow we have school.", 0),
    ("School starts tomorrow at 8 AM.", 0),
    ("We have a holiday next Monday.", 0),
    ("Your class starts at 9 AM tomorrow.", 0),
    ("Please bring your textbooks to school.", 0),
    ("Tomorrow is our regular school day.", 0),
    ("The teacher shared today's homework.", 0),
    ("Our school assembly starts at 8 AM.", 0),
    ("Your homework submission is due tomorrow.", 0),
    ("Your school bus will arrive at 7:30 AM.", 0),
    ("The science exam is on Monday.", 0),
    ("Please submit your assignment tomorrow.", 0),
    ("Your library book is due next week.", 0),
    ("The parent meeting starts at 10 AM.", 0),
    ("Your sports practice is cancelled today.", 0),
    ("Happy birthday! Have a wonderful day.", 0),
    ("I will meet you at the library.", 0),
    ("The football match starts at 5 PM.", 0),
    ("Please call me when you are free.", 0),
    ("The school trip is scheduled for Friday.", 0),

    # Suspicious messages
    ("Share your OTP immediately to claim your reward.", 1),
    ("Send your password to receive your prize.", 1),
    ("URGENT! Pay now to receive your free gift.", 1),
    ("Your account will be suspended. Verify immediately.", 1),
    ("You are a winner. Claim now!", 1),
    ("Click this unknown link to claim a cash prize.", 1),
    ("Provide your PIN to receive your reward.", 1),
    ("Send money now to unlock your prize.", 1),
    ("Your bank account will be blocked. Act now.", 1),
    ("Share your login details immediately.", 1),
    ("Pay a fee now to receive your reward.", 1),
    ("Your account is at risk. Verify immediately.", 1),
    ("Claim your free gift immediately.", 1),
    ("Transfer money now to receive your prize.", 1),
    ("Send your OTP to verify your account.", 1),
    ("You have won a cashback reward. Claim now!", 1),
    ("Click this link to claim your free gift.", 1),
    ("Your account will close. Act immediately.", 1),
    ("Provide your password to unlock your reward.", 1),
    ("Urgent! Send money to claim your prize.", 1)
]

X_train = [item[0] for item in messages]
y_train = [item[1] for item in messages]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y_train)


# ==========================================
# 3. IMPROVED RULE-BASED DETECTION
# ==========================================

import re

def analyse_message(message):

    text = " ".join(message.lower().split())
    reasons = []

    # 1. Sensitive information requests
    credential_patterns = [
        r"\bshare\b.{0,35}\b(otp|password|pin|passcode)\b",
        r"\bsend\b.{0,35}\b(otp|password|pin|passcode)\b",
        r"\b(provide|reveal|disclose)\b.{0,35}\b(otp|password|pin|passcode)\b",
        r"\b(login details|banking details|card details)\b",
        r"\b(enter|reply with)\b.{0,35}\b(otp|verification code)\b"
    ]

    credential_found = any(
        re.search(pattern, text) for pattern in credential_patterns
    )

    # 2. Money transfer requests
    money_patterns = [
        r"\b(send|transfer|deposit|pay)\b.{0,50}\b(money|cash|rupees|rs\.?|inr)\b",
        r"\b(send|transfer|deposit|pay)\b.{0,35}₹\s?\d+",
        r"\b(send|transfer|deposit|pay)\b.{0,35}\b\d+\b",
        r"\b(pay a fee|processing fee|transfer money)\b"
    ]

    money_found = any(
        re.search(pattern, text) for pattern in money_patterns
    )

    # 3. Account threats
    account_patterns = [
        r"\baccount\b.{0,35}\b(closed|blocked|suspended|disabled)\b",
        r"\b(account is at risk|account will close)\b",
        r"\b(sim|bank account)\b.{0,35}\b(blocked|suspended|stopped)\b"
    ]

    account_found = any(
        re.search(pattern, text) for pattern in account_patterns
    )

    # 4. Rewards and prizes
    reward_patterns = [
        r"\b(prize|reward|cashback|free gift|winner|lottery|voucher)\b"
    ]

    reward_found = any(
        re.search(pattern, text) for pattern in reward_patterns
    )

    # 5. Urgent pressure
    pressure_patterns = [
        r"\burgent(ly)?\b",
        r"\bimmediately\b",
        r"\basap\b",
        r"\bright away\b",
        r"\bact now\b",
        r"\bclaim now\b",
        r"\bwithin \d+ minutes?\b",
        r"\bexpires soon\b"
    ]

    pressure_found = any(
        re.search(pattern, text) for pattern in pressure_patterns
    )

    # 6. Suspicious link language
    link_patterns = [
        r"\bclick\b.{0,40}\b(link|url)\b",
        r"\b(open|visit)\b.{0,40}\b(link|url)\b",
        r"\blink below\b",
        r"\bunknown link\b"
    ]

    unknown_link_found = any(
        re.search(pattern, text) for pattern in link_patterns
    )

    # 7. Phone number or payment destination
    phone_found = bool(
        re.search(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", text)
    )

    upi_found = bool(
        re.search(r"\b[\w.-]+@[\w.-]+\b", text)
    )

    payment_destination_found = (
        phone_found or upi_found
    )

    # 8. Explain detected warning signs
    if credential_found:
        reasons.append(
            "The message requests sensitive information such as an OTP, password or PIN."
        )

    if money_found:
        reasons.append(
            "The message contains a possible money transfer or payment request."
        )

    if payment_destination_found and money_found:
        reasons.append(
            "A payment request is accompanied by a phone number or payment identifier."
        )

    if account_found and pressure_found:
        reasons.append(
            "The message combines an account threat with urgent pressure."
        )

    if reward_found and pressure_found:
        reasons.append(
            "The message combines a prize or reward with urgent pressure."
        )

    if unknown_link_found:
        reasons.append(
            "The message encourages clicking or opening a potentially unknown link."
        )

    if payment_destination_found and not money_found:
        reasons.append(
            "The message contains a phone number or payment identifier. This alone does not establish fraud."
        )

    # 9. Final decision
    suspicious = (
        credential_found
        or (money_found and (pressure_found or reward_found or account_found))
        or (account_found and pressure_found)
        or (reward_found and pressure_found)
        or unknown_link_found
        or (money_found and payment_destination_found)
    )

    # 10. Risk classification
    if credential_found:
        risk = "HIGH"
    elif suspicious:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return suspicious, risk, reasons



# ==========================================
# 4. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #071426,
        #0B1D32,
        #102B40
    );
    color: white;
}

h1, h2, h3 {
    color: #43D9C5 !important;
}

p, label, li {
    color: #E2EAF2;
}

div[data-testid="stMetric"] {
    background: #132B40;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #24516A;
}

.stButton > button {
    background-color: #13B8A6;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #0E9387;
    color: white;
}

div[data-testid="stTextArea"] textarea {
    background-color: #10283D;
    color: white;
    border: 1px solid #26758A;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. HEADER
# ==========================================

st.markdown(
    "<h1 style='text-align:center;'>🛡️ ScamShield AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center; color:#A9C9D8;'>"
    "THINK BEFORE YOU CLICK"
    "</h4>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "Your digital safety analysis assistant"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# 6. NAVIGATION
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Message Analyser",
    "🧪 Testing Dashboard",
    "📊 Visual Analytics",
    "ℹ️ About ScamShield"
])


# ==========================================
# 7. MESSAGE ANALYSER
# ==========================================

with tab1:

    left, right = st.columns([2, 1])

    with left:

        st.subheader("🔍 Analyse a Message")

        user_message = st.text_area(
            "Enter a fictional or non-sensitive message:",
            height=180,
            placeholder="Paste a fictional example here..."
        )

        analyse_button = st.button(
            "🔎 ANALYSE MESSAGE",
            use_container_width=True
        )

    with right:

        st.subheader("🛡️ About the analyser")

        st.write(
            "ScamShield looks for selected suspicious "
            "message patterns and provides safety guidance."
        )

        st.info(
            "Never enter real OTPs, passwords, PINs or "
            "private account information."
        )

    if analyse_button:

        if not user_message.strip():

            st.warning("Please enter a message first.")

        else:

            suspicious, risk, reasons = analyse_message(
                user_message
            )

            ml_vector = vectorizer.transform([user_message])

            ml_prediction = model.predict(ml_vector)[0]

            ml_probability = model.predict_proba(
                ml_vector
            )[0][1]

            st.divider()
            st.subheader("📋 Analysis Results")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Risk Level",
                    risk
                )

            with c2:
                st.metric(
                    "Warning Signs",
                    len(reasons)
                )

            with c3:
                ml_label = (
                    "Suspicious"
                    if ml_prediction == 1
                    else "No suspicious pattern"
                )

                st.metric(
                    "ML Model Signal",
                    ml_label
                )

            if suspicious:

                st.error("⚠️ SUSPICIOUS MESSAGE")

            else:

                st.success("✅ NOT FLAGGED")

            st.caption(
                "NOT FLAGGED does not mean that a message is safe."
            )

            st.subheader("🚩 Detected Warning Signs")

            if reasons:

                for reason in reasons:
                    st.warning(reason)

            else:

                st.info(
                    "No patterns covered by the current rules were detected."
                )

            st.subheader("🧠 Machine Learning Signal")

            st.write(
                f"Experimental model's suspicious-class score: "
                f"{ml_probability * 100:.1f}%"
            )

            st.caption(
                "This score is not a verified probability of a scam. "
                "The ML model is separate from the primary rule-based decision."
            )

            st.subheader("💡 Safety Recommendations")

            st.write("- Never share your OTP, PIN or password.")
            st.write("- Do not rush into sending money.")
            st.write("- Verify requests through official channels.")
            st.write("- Be careful with unexpected links.")
            st.write("- Ask a trusted adult if you are unsure.")

            st.info(
                "ScamShield AI is an experimental educational prototype. "
                "It may miss suspicious messages or flag ordinary messages."
            )


# ==========================================
# 8. TESTING DASHBOARD
# ==========================================

test_data = [

    # NORMAL MESSAGES
    ("The science exam is on Monday.", 0),
    ("Your school bus arrives at 7:30 AM.", 0),
    ("Please submit your homework tomorrow.", 0),
    ("Your library book is due next week.", 0),
    ("The parent meeting starts at 10 AM.", 0),
    ("Your sports practice is cancelled today.", 0),
    ("Happy birthday! Have a wonderful day.", 0),
    ("The school annual day is next Friday.", 0),
    ("Please bring your notebook to class.", 0),
    ("Your assignment has been checked.", 0),
    ("The football match starts at 5 PM.", 0),
    ("Your parcel has been delivered.", 0),
    ("Thank you for attending the meeting.", 0),
    ("Your class timetable has been updated.", 0),
    ("The science exhibition is tomorrow.", 0),
    ("Your appointment is confirmed for Monday.", 0),
    ("Your electricity bill is available online.", 0),
    ("Your payment receipt is ready.", 0),
    ("Your order has been shipped.", 0),
    ("The school will remain closed tomorrow.", 0),
    ("Please call me when you are free.", 0),
    ("Your bus pass expires next month.", 0),
    ("The result will be announced next week.", 0),
    ("Your registration was completed successfully.", 0),
    ("Thank you for your feedback.", 0),

    # OTP AND PASSWORD
    ("Share your OTP immediately to claim your reward.", 1),
    ("Send your password to receive your prize.", 1),
    ("Give us your banking PIN now.", 1),
    ("Reply with your OTP to verify your account.", 1),
    ("Send your login details to avoid account suspension.", 1),
    ("Your account is blocked. Share your password now.", 1),
    ("Provide your card PIN immediately.", 1),
    ("We need your OTP to release your cashback.", 1),
    ("Send your verification code to claim the reward.", 1),
    ("Your account will close unless you share your OTP.", 1),

    # MONEY TRANSFERS
    ("Urgent! Send Rs 500 immediately to receive your reward.", 1),
    ("Transfer ₹1000 now to unlock your prize.", 1),
    ("Pay now to receive your free gift.", 1),
    ("Deposit Rs 250 urgently to confirm your winning prize.", 1),
    ("Send money immediately to avoid account closure.", 1),
    ("Transfer INR 2000 to claim your cashback.", 1),
    ("Pay a processing fee now to receive your reward.", 1),
    ("Send 500 rupees to this number urgently.", 1),
    ("You must transfer money immediately to verify your account.", 1),
    ("Pay now or your reward will expire.", 1),
    ("Send Rs 750 within 5 minutes to claim your prize.", 1),
    ("Deposit money immediately to activate your reward.", 1),
    ("Transfer ₹1500 right away to receive your gift.", 1),
    ("Pay a fee urgently to unlock your account.", 1),
    ("Send 1000 to this number 9876543210 urgently.", 1),

    # PRIZES AND REWARDS
    ("Congratulations! You have won a free iPhone.", 1),
    ("You are a winner. Claim now!", 1),
    ("You have received a surprise cash prize.", 1),
    ("Claim your free gift immediately.", 1),
    ("You won a lottery. Contact us now.", 1),
    ("Your reward is waiting. Act now.", 1),
    ("You have been selected for a special prize.", 1),
    ("Claim your cashback before it expires.", 1),
    ("Congratulations! Receive your free shopping voucher.", 1),
    ("You are today's lucky winner. Claim your prize now.", 1),

    # UNKNOWN LINKS
    ("Click this unknown link to claim a cash prize.", 1),
    ("Open this link immediately to verify your account.", 1),
    ("Click the link below to receive your reward.", 1),
    ("Visit this unknown link to unlock your prize.", 1),
    ("Your account is blocked. Click this link now.", 1),
    ("Click this link urgently to claim your cashback.", 1),
    ("Open this link to prevent account suspension.", 1),
    ("Click this unknown link to receive a free gift.", 1),
    ("Verify your details through this link immediately.", 1),
    ("Act now! Click this link to claim your reward.", 1),

    # IMPERSONATION
    ("This is your bank. Share your OTP immediately.", 1),
    ("Your bank account will be suspended. Verify now.", 1),
    ("We are from customer support. Send your password.", 1),
    ("Your parcel is on hold. Pay a fee immediately.", 1),
    ("Your electricity connection will be stopped. Pay now.", 1),
    ("Government reward approved. Share your bank details.", 1),
    ("Your account has unusual activity. Send your PIN.", 1),
    ("This is the security team. Provide your login details.", 1),
    ("Your SIM will be blocked. Verify your OTP now.", 1),
    ("Your payment failed. Click this unknown link immediately.", 1),

    # EDGE CASES
    ("I paid the school trip fee yesterday.", 0),
    ("My friend sent me Rs 500 yesterday.", 0),
    ("Please transfer the project files to my laptop.", 0),
    ("The winner of the football match was our school.", 0),
    ("Our teacher said the exam is urgent.", 0),
    ("I received a reward for completing the school activity.", 0),
    ("The bank sent me a transaction notification.", 0),
    ("Please send the homework link to our class group.", 0),
    ("I need to pay my electricity bill this week.", 0),
    ("My account is working normally. Thank you.", 0),  
    # EXTRA TEST CASES
    ("Your account needs immediate verification. Send your OTP.", 1),
    ("Transfer Rs 300 now to unlock your account.", 1),
    ("You have won a shopping voucher. Claim immediately.", 1),
    ("Click this unknown link to update your banking details.", 1),
    ("Your delivery is delayed. Pay a fee urgently.", 1),
    ("Please bring your science textbook tomorrow.", 0),
    ("The school football tournament begins next month.", 0),
    ("I transferred money for my school fees yesterday.", 0),
    ("Your appointment reminder is scheduled for 4 PM.", 0),
    ("The teacher shared a new homework assignment.", 0),
]

actual_labels = []
predicted_labels = []
results = []

for test_message, expected in test_data:

    prediction, risk, reasons = analyse_message(
        test_message
    )

    predicted = 1 if prediction else 0

    actual_labels.append(expected)
    predicted_labels.append(predicted)

    results.append({
        "Test Message": test_message,
        "Expected": (
            "SUSPICIOUS" if expected == 1 else "NORMAL"
        ),
        "Predicted": (
            "SUSPICIOUS" if predicted == 1 else "NORMAL"
        ),
        "Result": (
            "CORRECT" if expected == predicted else "INCORRECT"
        )
    })


accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1]
).ravel()


with tab2:

    st.title("🧪 TESTING DASHBOARD")

    st.write(
        "Evaluate the prototype using fictional messages "
        "with manually assigned expected labels."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy * 100:.1f}%")
    c2.metric("Precision", f"{precision * 100:.1f}%")
    c3.metric("Recall", f"{recall * 100:.1f}%")
    c4.metric("F1 Score", f"{f1 * 100:.1f}%")

    st.divider()

    st.subheader("🎯 Confusion Matrix")

    m1, m2 = st.columns(2)

    m1.metric("Correct Normal", tn)
    m2.metric("Correct Suspicious", tp)

    m3, m4 = st.columns(2)

    m3.metric("False Alarms", fp)
    m4.metric("Missed Warnings", fn)

    st.divider()

    st.subheader("📋 Individual Test Results")

    results_df = pd.DataFrame(results)

    st.subheader("📋 Individual Test Results")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    missed_messages = results_df[
        (results_df["Expected"] == "SUSPICIOUS") &
        (results_df["Predicted"] == "NORMAL")
    ]

    st.subheader("🚨 Missed Suspicious Messages")

    st.dataframe(
        missed_messages,
        use_container_width=True,
        hide_index=True
    )
        "These results come from a small, manually created "
        "demonstration dataset. They do not establish "
        "real-world scam detection accuracy."
    )


# ==========================================
# 9. VISUAL ANALYTICS
# ==========================================

with tab3:

    st.title("📈 VISUAL ANALYTICS")

    st.write(
        "Explore the prototype's results on the demonstration dataset."
    )

    st.subheader("🎯 Classification Breakdown")

    chart_data = pd.DataFrame(
        {
            "Number of Messages": [
                tn,
                fp,
                fn,
                tp
            ]
        },
        index=[
            "Correct Normal",
            "False Alarms",
            "Missed Warnings",
            "Correct Suspicious"
        ]
    )

    st.bar_chart(chart_data)

    st.subheader("📊 Correct vs Incorrect")

    correct = tn + tp
    incorrect = fp + fn

    outcome_data = pd.DataFrame(
        {
            "Messages": [
                correct,
                incorrect
            ]
        },
        index=[
            "Correct Predictions",
            "Incorrect Predictions"
        ]
    )

    st.bar_chart(outcome_data)

    st.subheader("📨 Test Dataset Composition")

    class_data = pd.DataFrame(
        {
            "Messages": [
                sum(actual_labels),
                len(actual_labels) - sum(actual_labels)
            ]
        },
        index=[
            "Suspicious",
            "Normal"
        ]
    )

    st.bar_chart(class_data)

    st.divider()

    st.subheader("📥 Download Test Report")

    csv_report = results_df.to_csv(index=False)

    st.download_button(
        label="⬇️ DOWNLOAD CSV REPORT",
        data=csv_report,
        file_name="scamshield_test_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Charts use the current manually labelled demonstration dataset. "
        "They are not evidence of real-world detection accuracy."
    )


# ==========================================
# 10. ABOUT THE PROJECT
# ==========================================

with tab4:

    st.title("ℹ️ ABOUT ScamShield AI")

    st.subheader("Our Mission")

    st.write(
        "ScamShield AI aims to help people recognise suspicious "
        "message patterns and make safer digital decisions."
    )

    st.subheader("Technology Used")

    st.markdown("""
    - **Python:** Main programming language
    - **Streamlit:** Web application interface
    - **Scikit-learn:** Machine learning tools
    - **TF-IDF:** Text feature extraction
    - **Logistic Regression:** Experimental text classifier
    - **Pandas:** Data handling and CSV reports
    """)

    st.subheader("How It Works")

    st.markdown("""
    1. A user enters a message.
    2. Rule-based checks look for selected warning patterns.
    3. A separate ML model produces an additional signal.
    4. The application displays findings and safety suggestions.
    """)

    st.subheader("Current Limitations")

    st.warning("""
    - The training dataset is small and manually created.
    - The main detection system uses predefined rules.
    - The ML signal is experimental.
    - Some suspicious messages may not be detected.
    - Some ordinary messages may be incorrectly flagged.
    - The demonstration metrics are not representative of real-world performance.
    """)

    st.subheader("Future Improvements")

    st.markdown("""
    - Expand the dataset using carefully reviewed examples.
    - Add more Indian scam-message patterns.
    - Explore Telugu and Hindi language support.
    - Conduct independent testing.
    - Improve false-positive and missed-warning rates.
    - Strengthen privacy protections.
    """)

    st.divider()

    st.markdown(
        "<h3 style='text-align:center;'>"
        "🛡️ ScamShield AI"
        "</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;'>"
        "THINK BEFORE YOU CLICK."
        "</p>",
        unsafe_allow_html=True
    )

    st.caption(
        "Experimental educational prototype | hridhAI Hackathon 2026"
    )
