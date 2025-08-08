import streamlit as st
from PIL import Image
import pandas as pd
import os

st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman", ["Visualisasi Hasil Pengujian", "Perbandingan Model"])

if halaman == "Visualisasi Hasil Pengujian":
    st.title("Visualisasi Hasil Pengujian")

    dataset = st.selectbox("Pilih Dataset", ["Dataset 1", "Dataset 2", "Dataset 3"])
    model = st.selectbox("Pilih Model", ["EfficientNet-B0", "EfficientNet-B7", "ConvNeXt-Tiny", "ConvNeXt-XL"])

    dataset_id = dataset.replace(" ", "_").lower()
    model_id = model.replace("-", "").replace(" ", "_").lower()

    st.subheader("Grafik Training & Validation")
    vis_path = f"visualisasi/{dataset_id}_{model_id}.png"
    if os.path.exists(vis_path):
        st.image(vis_path, caption="Training & Validation Graph", use_container_width=True)
    else:
        st.warning("Grafik training belum tersedia.")

    st.subheader("Confusion Matrix")
    cm_path = f"confusion_matrix/{dataset_id}_{model_id}.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning("Confusion Matrix belum tersedia.")

    st.subheader("Classification Report")
    report_path = f"classification_report/{dataset_id}_{model_id}.csv"
    if os.path.exists(report_path):
        df_report = pd.read_csv(report_path, index_col=0)
        st.dataframe(df_report)
    else:
        st.warning("Classification Report belum tersedia.")

elif halaman == "Perbandingan Model":
    st.title("Tabel Perbandingan Model")

    data_precison= {
        "Model": ["EfficientNet-B0", "ConvNeXt-Tiny", "EfficientNet-B7",  "ConvNeXt-XL"],
        "Dataset 1": [0.97, 0.97, 0.93, 0.99],
        "Dataset 2": [0.98, 0.97, 0.97, 1.00],
        "Dataset 3": [0.97, 0.97, 0.97, 0.95]
    }

    data_recall = {
        "Model": ["EfficientNet-B0", "ConvNeXt-Tiny", "EfficientNet-B7",  "ConvNeXt-XL"],
        "Dataset 1": [0.96, 0.97, 0.93, 0.99],
        "Dataset 2": [0.97, 0.97, 0.97, 1.00],
        "Dataset 3": [0.97, 0.97, 0.95, 0.99]
    }

    data_F1 = {
        "Model": ["EfficientNet-B0", "ConvNeXt-Tiny", "EfficientNet-B7",  "ConvNeXt-XL"],
        "Dataset 1": [0.96, 0.97, 0.93, 0.99],
        "Dataset 2": [0.97, 0.97, 0.97, 1.00],
        "Dataset 3": [0.97, 0.97, 0.95, 0.99]
    }

    data_akurasi = {
        "Model": ["EfficientNet-B0", "ConvNeXt-Tiny", "EfficientNet-B7",  "ConvNeXt-XL"],
        "Dataset 1": [0.96, 0.97, 0.93, 0.99],
        "Dataset 2": [0.97, 0.97, 0.97, 1.00],
        "Dataset 3": [0.97, 0.97, 0.95, 0.99]
    }

    st.subheader("Perbandingan Macro Avg Precision")
    st.dataframe(pd.DataFrame(data_precison))

    st.subheader("Perbandingan Macro Avg Recall")
    st.dataframe(pd.DataFrame(data_recall))

    st.subheader("Perbandingan Macro Avg F1-Score")
    st.dataframe(pd.DataFrame(data_F1))

    st.subheader("Perbandingan Macro Avg Accuracy")
    st.dataframe(pd.DataFrame(data_akurasi))

