
import json
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from database import (
    init_db,
    SessionLocal,
    save_student,
    get_students,
    delete_student,
    get_all_users,
    delete_user,
    log_action,
    get_audit_logs
)

from authentication import login, create_user
from classifier import predict_students


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

METADATA_FILE = MODEL_DIR / "model_metadata.json"
PDF_FILE = REPORT_DIR / "Student_Model_Evaluation_Report.pdf"

REPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Student Classification System",
    page_icon="🎓",
    layout="wide"
)


# ==========================================================
# DATABASE INITIALIZATION
# ==========================================================

@st.cache_resource
def initialize_database():
    init_db()
    return True


initialize_database()


# ==========================================================
# SESSION INITIALIZATION
# ==========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "pending_student" not in st.session_state:
    st.session_state.pending_student = None

if "bulk_result" not in st.session_state:
    st.session_state.bulk_result = None


# ==========================================================
# GENERAL FUNCTIONS
# ==========================================================

def get_metadata():

    if not METADATA_FILE.exists():
        return {}

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return {}


def get_prediction_result(result):

    if isinstance(result, pd.DataFrame):

        if result.empty:
            raise ValueError("Model returned an empty prediction.")

        return result.iloc[0].to_dict()

    if isinstance(result, pd.Series):
        return result.to_dict()

    if isinstance(result, dict):
        return result

    if isinstance(result, (list, tuple)):

        if not result:
            raise ValueError("Model returned no prediction.")

        if isinstance(result[0], dict):
            return result[0]

        return {"Prediction": result[0]}

    return {"Prediction": result}


def get_prediction_label(result):

    possible_keys = [
        "predicted_risk",
        "prediction",
        "Prediction",
        "TARGET_RISK",
        "risk",
        "class"
    ]

    for key in possible_keys:

        if key in result:
            return str(result[key])

    return str(next(iter(result.values())))


def dataframe_download(df, filename):

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download {filename}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        width="stretch"
    )


def create_metrics_table(metadata):

    report = metadata.get("classification_report", {})

    rows = []

    if not isinstance(report, dict):
        return pd.DataFrame()

    for label, values in report.items():

        if not isinstance(values, dict):
            continue

        if label in ("accuracy", "macro avg", "weighted avg"):
            continue

        rows.append({
            "Class": str(label),
            "Precision": values.get("precision", 0),
            "Recall": values.get("recall", 0),
            "F1 Score": values.get("f1-score", 0),
            "Support": values.get("support", 0)
        })

    for label in ("macro avg", "weighted avg"):

        values = report.get(label)

        if isinstance(values, dict):

            rows.append({
                "Class": label,
                "Precision": values.get("precision", 0),
                "Recall": values.get("recall", 0),
                "F1 Score": values.get("f1-score", 0),
                "Support": values.get("support", 0)
            })

    return pd.DataFrame(rows)


def create_confusion_dataframe(metadata):

    matrix = metadata.get("confusion_matrix", [])
    classes = metadata.get("classes", [])

    if not matrix:
        return pd.DataFrame()

    df = pd.DataFrame(matrix)

    if classes and len(classes) == len(df.columns):
        df.columns = [str(x) for x in classes]

    if classes and len(classes) == len(df.index):
        df.index = [str(x) for x in classes]

    df.index.name = "Actual / Predicted"

    return df


# ==========================================================
# LOGIN PAGE
# ==========================================================

def login_page():

    st.title("🎓 Student Classification System")
    st.subheader("Login to Your Account")

    with st.form("login_form"):

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Login",
            width="stretch"
        )

        if submitted:

            db = SessionLocal()

            try:

                user = login(
                    db,
                    username.strip(),
                    password
                )

                if user:

                    st.session_state.authenticated = True
                    st.session_state.username = user.username
                    st.session_state.role = user.role

                    log_action(
                        db,
                        user.username,
                        "Login",
                        "User logged in"
                    )

                    db.commit()

                    st.success("Login successful.")
                    st.rerun()

                else:

                    st.error("Invalid username or password.")

            except Exception as error:

                db.rollback()
                st.error(f"Login error: {error}")

            finally:
                db.close()


# ==========================================================
# DASHBOARD
# ==========================================================

def dashboard_page():

    st.title("📊 Dashboard")

    db = SessionLocal()

    try:

        students = get_students(db)

        total_students = len(students)

        risk_counts = {}

        for student in students:

            risk = str(student.predicted_risk)

            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Student Records",
            total_students
        )

        col2.metric(
            "High Risk",
            risk_counts.get("High", 0)
        )

        col3.metric(
            "Low Risk",
            risk_counts.get("Low", 0)
        )

        if risk_counts:

            st.subheader("Student Risk Distribution")

            chart_df = pd.DataFrame(
                list(risk_counts.items()),
                columns=["Risk Category", "Students"]
            )

            st.bar_chart(
                chart_df.set_index("Risk Category"),
                width="stretch"
            )

        else:

            st.info("No student records available.")

    except Exception as error:

        st.error(f"Dashboard error: {error}")

    finally:
        db.close()


# ==========================================================
# INDIVIDUAL PREDICTION
# ==========================================================

def individual_prediction_page():

    st.title("👤 Individual Student Prediction")

    metadata = get_metadata()

    if not metadata:
        st.error("Model metadata not found. Train your model first.")
        return

    features = metadata.get("features", [])
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])

    if not features:
        st.error("Model features are unavailable.")
        return

    st.subheader("Enter Student Information")

    values = {}

    columns = st.columns(2)

    for index, feature in enumerate(features):

        with columns[index % 2]:

            if feature in numeric_features:

                value = st.number_input(
                    str(feature),
                    value=0.0,
                    format="%.4f",
                    key=f"individual_{feature}"
                )

            elif feature in categorical_features:

                value = st.text_input(
                    str(feature),
                    key=f"individual_{feature}"
                )

            else:

                value = st.text_input(
                    str(feature),
                    key=f"individual_{feature}"
                )

            values[feature] = value

    if st.button(
        "Predict Student Risk",
        width="stretch"
    ):

        try:

            input_df = pd.DataFrame([values])

            prediction = predict_students(input_df)

            result = get_prediction_result(prediction)

            risk = get_prediction_label(result)

            st.session_state.pending_student = {
                "input": values,
                "prediction": risk
            }

            st.success(f"Predicted Risk: {risk}")

        except Exception as error:

            st.error(f"Prediction failed: {error}")

    if st.session_state.pending_student:

        pending = st.session_state.pending_student

        st.write(
            "Prediction:",
            pending["prediction"]
        )

        if st.button(
            "Save Student Record",
            width="stretch"
        ):

            db = SessionLocal()

            try:

                save_student(
                    db,
                    pending["input"],
                    pending["prediction"],
                    st.session_state.username
                )

                log_action(
                    db,
                    st.session_state.username,
                    "Student Prediction",
                    f"Prediction saved: {pending['prediction']}"
                )

                db.commit()

                st.session_state.pending_student = None

                st.success("Student record saved successfully.")
                st.rerun()

            except Exception as error:

                db.rollback()
                st.error(f"Unable to save record: {error}")

            finally:
                db.close()


# ==========================================================
# BULK PREDICTION
# ==========================================================

def bulk_prediction_page():

    st.title("📁 Bulk Student Prediction")

    st.write(
        "Upload a CSV file containing the input features "
        "used during model training."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is None:
        return

    try:

        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(
            input_df,
            width="stretch"
        )

        if st.button(
            "Run Bulk Prediction",
            width="stretch"
        ):

            predictions = predict_students(input_df)

            if isinstance(predictions, pd.DataFrame):

                if len(predictions) != len(input_df):
                    raise ValueError(
                        "Prediction count does not match input rows."
                    )

                result_df = input_df.copy()

                for column in predictions.columns:
                    result_df[column] = predictions[column].values

            elif isinstance(predictions, pd.Series):

                if len(predictions) != len(input_df):
                    raise ValueError(
                        "Prediction count does not match input rows."
                    )

                result_df = input_df.copy()
                result_df["Prediction"] = predictions.values

            else:

                prediction_values = list(predictions)

                if len(prediction_values) != len(input_df):
                    raise ValueError(
                        "Prediction count does not match input rows."
                    )

                result_df = input_df.copy()
                result_df["Prediction"] = prediction_values

            st.session_state.bulk_result = result_df

            st.success("Bulk prediction completed.")

    except Exception as error:

        st.error(f"Bulk prediction failed: {error}")

    if st.session_state.bulk_result is not None:

        result_df = st.session_state.bulk_result

        st.subheader("Prediction Results")

        st.dataframe(
            result_df,
            width="stretch"
        )

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            data=csv_data,
            file_name="student_predictions.csv",
            mime="text/csv",
            width="stretch"
        )


# ==========================================================
# STUDENT RECORDS
# ==========================================================

def student_records_page():

    st.title("📚 Student Records")

    db = SessionLocal()

    try:

        students = get_students(db)

        if not students:

            st.info("No student records found.")
            return

        rows = []

        for student in students:

            try:
                input_data = json.loads(student.input_data or "{}")
            except Exception:
                input_data = {}

            row = {
                "ID": student.id,
                "Predicted Risk": student.predicted_risk,
                "Created By": student.created_by,
                "Created At": str(student.created_at)
            }

            if isinstance(input_data, dict):
                row.update(input_data)

            rows.append(row)

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            width="stretch"
        )

        dataframe_download(
            df,
            "student_records.csv"
        )

        st.divider()

        st.subheader("Delete Student Record")

        student_ids = [
            student.id for student in students
        ]

        selected_id = st.selectbox(
            "Select Student Record ID",
            student_ids
        )

        confirm = st.checkbox(
            "I confirm that I want to delete this record."
        )

        if st.button(
            "Delete Selected Record",
            disabled=not confirm,
            width="stretch"
        ):

            delete_student(db, selected_id)

            log_action(
                db,
                st.session_state.username,
                "Delete Student",
                f"Deleted student record ID {selected_id}"
            )

            db.commit()

            st.success("Student record deleted.")
            st.rerun()

    except Exception as error:

        db.rollback()
        st.error(f"Student records error: {error}")

    finally:
        db.close()


# ==========================================================
# MODEL EVALUATION REPORT - PDF
# ==========================================================

def evaluation_report_page():

    st.title("📈 Model Evaluation Report")

    metadata = get_metadata()

    if not metadata:

        st.error(
            "Model metadata not found. "
            "Run python train_model.py first."
        )

        return

    accuracy = float(metadata.get("accuracy", 0) or 0)
    cv_f1 = float(metadata.get("cv_macro_f1", 0) or 0)
    test_f1 = float(metadata.get("test_macro_f1", 0) or 0)

    # OVERALL METRICS

    st.subheader("Overall Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Test Macro F1",
        f"{test_f1 * 100:.2f}%"
    )

    col3.metric(
        "Cross Validation F1",
        f"{cv_f1 * 100:.2f}%"
    )

    st.divider()

    # MODEL INFORMATION

    st.subheader("Model Information")

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Best Model",
        str(metadata.get("best_model", "N/A"))
    )

    info2.metric(
        "Dataset Records",
        str(metadata.get("records", "N/A"))
    )

    info3.metric(
        "Features",
        str(metadata.get("feature_count", "N/A"))
    )

    st.divider()

    # CLASSIFICATION REPORT

    st.subheader("Precision, Recall and F1 Score")

    metrics_df = create_metrics_table(metadata)

    if not metrics_df.empty:

        st.dataframe(
            metrics_df,
            width="stretch",
            hide_index=True
        )

        dataframe_download(
            metrics_df,
            "precision_recall_f1_report.csv"
        )

    else:

        st.warning(
            "Class-wise classification report is unavailable. "
            "Retrain your model with classification_report saved."
        )

    st.divider()

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    matrix_df = create_confusion_dataframe(metadata)

    if not matrix_df.empty:

        st.dataframe(
            matrix_df,
            width="stretch"
        )

        st.bar_chart(
            matrix_df,
            width="stretch"
        )

        matrix_csv = matrix_df.to_csv().encode("utf-8")

        st.download_button(
            "Download Confusion Matrix CSV",
            data=matrix_csv,
            file_name="confusion_matrix.csv",
            mime="text/csv",
            width="stretch"
        )

    else:

        st.warning("Confusion matrix is unavailable.")

    st.divider()

    # EXPLANATION

    st.subheader("Evaluation Metrics Explanation")

    st.markdown("""
    **Accuracy:** Percentage of total predictions that are correct.

    **Precision:** Percentage of predicted positive cases that are actually positive.

    **Recall:** Percentage of actual positive cases correctly identified.

    **F1 Score:** Harmonic mean of precision and recall.

    **Confusion Matrix:** Shows correct and incorrect predictions for each class.

    **Cross Validation F1:** Average F1 score across cross-validation folds.
    """)

    st.divider()

    # PDF GENERATION AND DOWNLOAD

    st.subheader("📄 PDF Evaluation Report")

    st.write(
        "Generate and download the complete model performance "
        "report in PDF format."
    )

    from generate_report import generate_pdf_report

    if st.button(
        "Generate PDF Report",
        width="stretch"
    ):

        try:

            generate_pdf_report()

            st.success(
                "PDF evaluation report generated successfully."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"PDF generation failed: {error}"
            )

    if PDF_FILE.exists():

        with open(PDF_FILE, "rb") as pdf_file:

            pdf_data = pdf_file.read()

        st.download_button(
            label="⬇️ Download Model Evaluation PDF",
            data=pdf_data,
            file_name="Student_Model_Evaluation_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )

    else:

        st.info(
            "Click Generate PDF Report to create your report."
        )


# ==========================================================
# MODEL INFORMATION
# ==========================================================

def model_information_page():

    st.title("🤖 Model Information")

    metadata = get_metadata()

    if not metadata:

        st.error("Model metadata not found.")
        return

    st.write(
        "**Model:**",
        metadata.get("best_model", "N/A")
    )

    st.write(
        "**Target:**",
        metadata.get("target", "N/A")
    )

    st.write(
        "**Dataset Records:**",
        metadata.get("records", "N/A")
    )

    st.write(
        "**Number of Features:**",
        metadata.get("feature_count", "N/A")
    )

    st.subheader("Input Features")

    features = metadata.get("features", [])

    if features:
        st.write(features)
    else:
        st.info("Feature information unavailable.")

    st.subheader("Class Labels")

    classes = metadata.get("classes", [])

    if classes:
        st.write(classes)
    else:
        st.info("Class information unavailable.")


# ==========================================================
# USER MANAGEMENT
# ==========================================================

def user_management_page():

    st.title("👥 User Management")

    if st.session_state.role != "admin":

        st.error("Only administrators can access this page.")
        return

    db = SessionLocal()

    try:

        st.subheader("Create New User")

        with st.form("create_user_form"):

            username = st.text_input("New Username")

            password = st.text_input(
                "Password",
                type="password"
            )

            role = st.selectbox(
                "Role",
                ["faculty", "admin"]
            )

            submitted = st.form_submit_button(
                "Create User",
                width="stretch"
            )

            if submitted:

                success, message = create_user(
                    db,
                    username.strip(),
                    password,
                    role
                )

                if success:

                    log_action(
                        db,
                        st.session_state.username,
                        "Create User",
                        f"Created {role} account: {username}"
                    )

                    db.commit()

                    st.success(message)
                    st.rerun()

                else:

                    db.rollback()
                    st.error(message)

        st.divider()

        st.subheader("Existing Users")

        users = get_all_users(db)

        if users:

            user_rows = []

            for user in users:

                user_rows.append({
                    "ID": user.id,
                    "Username": user.username,
                    "Role": user.role,
                    "Created At": str(user.created_at)
                })

            st.dataframe(
                pd.DataFrame(user_rows),
                width="stretch",
                hide_index=True
            )

            removable_users = [
                user for user in users
                if user.username != st.session_state.username
            ]

            if removable_users:

                selected_user = st.selectbox(
                    "Select User to Delete",
                    removable_users,
                    format_func=lambda user:
                        f"{user.username} ({user.role})"
                )

                confirm = st.checkbox(
                    "Confirm user deletion"
                )

                if st.button(
                    "Delete User",
                    disabled=not confirm,
                    width="stretch"
                ):

                    delete_user(
                        db,
                        selected_user.id
                    )

                    log_action(
                        db,
                        st.session_state.username,
                        "Delete User",
                        f"Deleted user: {selected_user.username}"
                    )

                    db.commit()

                    st.success("User deleted.")
                    st.rerun()

        else:

            st.info("No users found.")

    except Exception as error:

        db.rollback()
        st.error(f"User management error: {error}")

    finally:
        db.close()


# ==========================================================
# AUDIT LOGS
# ==========================================================

def audit_logs_page():

    st.title("📜 Audit Logs")

    if st.session_state.role != "admin":

        st.error("Only administrators can access audit logs.")
        return

    db = SessionLocal()

    try:

        logs = get_audit_logs(db)

        if not logs:

            st.info("No audit logs found.")
            return

        rows = []

        for log in logs:

            rows.append({
                "Username": log.username,
                "Action": log.action,
                "Details": log.detail,
                "Timestamp": str(log.timestamp)
            })

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )

        dataframe_download(
            df,
            "audit_logs.csv"
        )

    except Exception as error:

        st.error(f"Audit log error: {error}")

    finally:
        db.close()


# ==========================================================
# MAIN APPLICATION
# ==========================================================

def main():

    if not st.session_state.authenticated:

        login_page()
        return

    st.sidebar.title("🎓 Student Classification")

    st.sidebar.write(
        f"User: {st.session_state.username}"
    )

    st.sidebar.write(
        f"Role: {st.session_state.role}"
    )

    pages = [
        "Dashboard",
        "Individual Prediction",
        "Bulk Prediction",
        "Student Records",
        "Model Evaluation Report",
        "Model Information"
    ]

    if st.session_state.role == "admin":

        pages.extend([
            "User Management",
            "Audit Logs"
        ])

    selected_page = st.sidebar.radio(
        "Navigation",
        pages
    )

    if st.sidebar.button(
        "Logout",
        width="stretch"
    ):

        db = SessionLocal()

        try:

            log_action(
                db,
                st.session_state.username,
                "Logout",
                "User logged out"
            )

            db.commit()

        except Exception:
            db.rollback()

        finally:
            db.close()

        st.session_state.clear()
        st.rerun()

    if selected_page == "Dashboard":
        dashboard_page()

    elif selected_page == "Individual Prediction":
        individual_prediction_page()

    elif selected_page == "Bulk Prediction":
        bulk_prediction_page()

    elif selected_page == "Student Records":
        student_records_page()

    elif selected_page == "Model Evaluation Report":
        evaluation_report_page()

    elif selected_page == "Model Information":
        model_information_page()

    elif selected_page == "User Management":
        user_management_page()

    elif selected_page == "Audit Logs":
        audit_logs_page()


# ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()

