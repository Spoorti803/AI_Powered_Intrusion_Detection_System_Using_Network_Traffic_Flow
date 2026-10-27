
import os
import pickle
import warnings

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pickle(filename):
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, "rb") as file:
        return pickle.load(file)


# Load saved artifacts
@st.cache_resource
def load_model_artifacts():
    model = load_pickle("xgboost_multiclass.pkl")
    encoder = load_pickle("label_encoder.pkl")
    features = load_pickle("selected_features.pkl")
    return model, encoder, features


st.set_page_config(
    page_title="Intrusion detection Uisng XGBoost",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Intrusion Detection System")
st.caption("Network Flow Analysis using an XGBoost Multiclass Classifier")

try:
    model, label_encoder, selected_features = load_model_artifacts()
except Exception as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "Classification Report",
        "Feature Importance",
        "Traffic Prediction"
    ]
)


# Dashboard
if page == "Dashboard":
    st.header("Project Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model", "XGBoost")
        st.metric("Selected Features", len(selected_features))

    with col2:
        st.metric("Attack Categories", len(label_encoder.classes_))
        st.metric("Prediction Type", "Multiclass")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write(
        "This application uses a trained XGBoost multiclass classifier "
        "to classify network traffic based on network-flow features."
    )

    st.info(
        "Upload a CSV containing the required selected features to obtain "
        "predicted traffic categories. The model does not retrain when "
        "the application starts."
    )

    st.subheader("Supported Classes")

    st.write(", ".join(map(str, label_encoder.classes_)))


# Classification Report
elif page == "Classification Report":
    st.header("Model Classification Report")

    report_path = os.path.join(
        BASE_DIR,
        "classification_report.csv"
    )

    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path, index_col=0)

        st.dataframe(report_df, use_container_width=True)

        metric_rows = report_df.index.intersection(
            ["macro avg", "weighted avg"]
        )

        if len(metric_rows) > 0:
            st.subheader("Summary Metrics")
            st.dataframe(
                report_df.loc[metric_rows],
                use_container_width=True
            )

        if "f1-score" in report_df.columns:
            class_rows = report_df.index.difference(
                ["accuracy", "macro avg", "weighted avg"]
            )

            class_f1 = pd.to_numeric(
                report_df.loc[class_rows, "f1-score"],
                errors="coerce"
            ).dropna()

            if not class_f1.empty:
                st.subheader("F1-score by Class")

                fig, ax = plt.subplots(figsize=(11, 6))
                class_f1.sort_values().plot.barh(ax=ax)
                ax.set_xlabel("F1-score")
                ax.set_xlim(0, 1)
                ax.set_title("Classification Performance by Class")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

        # st.warning(
        #     "These are previously generated evaluation results. "
        #     "Rare attack classes have very few test samples, so their "
        #     "metrics should be interpreted cautiously."
        # )

    else:
        st.warning(
            "classification_report.csv was not found. "
            "Download it from Kaggle and place it beside app.py."
        )


# Feature Importance
elif page == "Feature Importance":
    st.header("Feature Importance")

    importance_path = os.path.join(
        BASE_DIR,
        "shap_feature_importance.csv"
    )

    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)

        st.subheader("Saved Feature-Importance Results")
        st.dataframe(importance_df, use_container_width=True)

        numeric_columns = importance_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            possible_names = [
                "feature",
                "Feature",
                "feature_name",
                "Feature Name"
            ]

            name_column = next(
                (
                    column for column in possible_names
                    if column in importance_df.columns
                ),
                None
            )

            if name_column is not None:
                plot_df = importance_df[
                    [name_column, numeric_columns[0]]
                ].dropna()

                plot_df = plot_df.sort_values(
                    numeric_columns[0]
                ).tail(20)

                fig, ax = plt.subplots(figsize=(10, 7))

                ax.barh(
                    plot_df[name_column].astype(str),
                    plot_df[numeric_columns[0]]
                )

                ax.set_xlabel(numeric_columns[0])
                ax.set_title("Top Feature Importance Values")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

    else:
        st.warning("Feature-importance CSV not found.")


# Traffic Prediction
elif page == "Traffic Prediction":
    st.header("Network Traffic Prediction")

    st.write(
        "Upload a CSV containing the selected network-flow features "
        "required by the saved model."
    )

    st.write(f"Required number of features: {len(selected_features)}")

    with st.expander("View required feature names"):
        st.write(selected_features)

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")
            st.dataframe(input_df.head(), use_container_width=True)

            missing_features = [
                feature for feature in selected_features
                if feature not in input_df.columns
            ]

            if missing_features:
                st.error(
                    "The uploaded CSV is missing required features: "
                    + ", ".join(missing_features)
                )

            else:
                input_features = input_df[selected_features].copy()

                for column in selected_features:
                    input_features[column] = pd.to_numeric(
                        input_features[column],
                        errors="coerce"
                    )

                invalid_rows = input_features.isna().any(axis=1)

                if invalid_rows.any():
                    st.error(
                        f"{invalid_rows.sum()} rows contain missing or "
                        "non-numeric feature values. Clean these rows "
                        "before making predictions."
                    )

                else:
                    if st.button("Predict Traffic Categories"):
                        predictions = model.predict(input_features)

                        predicted_labels = label_encoder.inverse_transform(
                            predictions.astype(int)
                        )

                        results_df = input_df.copy()
                        results_df["Predicted_Attack"] = predicted_labels

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(
                                input_features
                            )

                            results_df["Prediction_Confidence"] = (
                                probabilities.max(axis=1)
                            )

                        st.subheader("Prediction Results")
                        st.dataframe(
                            results_df.head(100),
                            use_container_width=True
                        )

                        st.subheader("Prediction Summary")
                        st.dataframe(
                            results_df["Predicted_Attack"]
                            .value_counts()
                            .rename_axis("Predicted Category")
                            .reset_index(name="Number of Flows"),
                            use_container_width=True
                        )

                        csv_data = results_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Prediction Results",
                            data=csv_data,
                            file_name="ids_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as error:
            st.error(f"Prediction failed: {error}")