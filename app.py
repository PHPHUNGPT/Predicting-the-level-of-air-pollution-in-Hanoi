import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import matplotlib.pyplot as plt

# Cài đặt tiêu đề cho ứng dụng
st.title("Dự đoán Nồng Độ Bụi Mịn PM2.5")

# Tải dữ liệu và mô hình
DATA_DIR = "resources"
MODELS_DIR = "models"

# Đọc danh sách các mô hình đã lưu
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
model_names = [f.replace("_optimized.joblib", "").replace("_", " ") for f in model_files]

# Đọc dữ liệu đặc trưng
cleaned_data = pd.read_csv(os.path.join(DATA_DIR, "X_values.csv"))
y_actual = pd.read_csv(os.path.join(DATA_DIR, "y_values.csv"))
feature_list = cleaned_data.columns.tolist()

# Lưu lịch sử dự đoán
if "history" not in st.session_state:
    st.session_state["history"] = []

# Tùy chỉnh giao diện bên trái
st.sidebar.header("Tùy chỉnh dự đoán")

# Người dùng chọn chế độ dự đoán
input_mode = st.sidebar.radio(
    "Chọn chế độ nhập liệu:",
    ["Nhập thủ công", "Dự đoán từ bộ dữ liệu ban đầu"]
)

# Người dùng chọn mô hình dự đoán
prediction_mode = st.sidebar.radio(
    "Chọn chế độ dự đoán:",
    ["Dự đoán bằng một mô hình", "Dự đoán bằng tất cả các mô hình"]
)

if prediction_mode == "Dự đoán bằng một mô hình":
    selected_model = st.sidebar.selectbox(
        "Chọn mô hình để dự đoán:", options=model_names
    )
else:
    selected_model = "Tất cả mô hình"

# Nếu chọn dự đoán từ bộ dữ liệu ban đầu
if input_mode == "Dự đoán từ bộ dữ liệu ban đầu":
    st.sidebar.header("Dự đoán từ dữ liệu ban đầu")
    selected_rows = st.sidebar.multiselect(
        "Chọn hàng dữ liệu để dự đoán:",
        options=cleaned_data.index.tolist(),
        default=cleaned_data.index.tolist()
    )
    input_values = cleaned_data.iloc[selected_rows].values
    selected_y_actual = y_actual.iloc[selected_rows]
else:
    # Nếu chọn nhập liệu thủ công
    st.sidebar.header("Nhập giá trị các đặc trưng")
    input_data = {}
    for feature in feature_list:
        input_data[feature] = st.sidebar.number_input(f"Nhập giá trị cho {feature}", value=0.0)
    input_values = np.array([list(input_data.values())])

# Dự đoán và hiển thị kết quả
st.header("Dự đoán PM2.5")
if st.sidebar.button("Dự đoán"):
    if prediction_mode == "Dự đoán bằng một mô hình":
        # Dự đoán với mô hình đã chọn
        model_file_path = os.path.join(MODELS_DIR, f"{selected_model.replace(' ', '_')}_optimized.joblib")
        if os.path.exists(model_file_path):
            model = joblib.load(model_file_path)
            try:
                predictions = model.predict(input_values)
                for i, prediction in enumerate(predictions):
                    st.success(f"Kết quả dự đoán PM2.5 với mô hình {selected_model} (Hàng {i}): {prediction:.2f}")
                    # Lưu lịch sử
                    st.session_state["history"].append({
                        "Model": selected_model,
                        **{feature: input_values[i][j] for j, feature in enumerate(feature_list)},
                        "Prediction": prediction
                    })
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}")
        else:
            st.error("Không tìm thấy mô hình đã chọn.")
    else:
        # Dự đoán với tất cả các mô hình
        predictions = []
        for model_name, model_file in zip(model_names, model_files):
            model_file_path = os.path.join(MODELS_DIR, model_file)
            if os.path.exists(model_file_path):
                model = joblib.load(model_file_path)
                try:
                    model_predictions = model.predict(input_values)
                    for i, prediction in enumerate(model_predictions):
                        predictions.append({"Model": model_name, "Row": i, "Prediction": prediction})
                        # Lưu lịch sử
                        st.session_state["history"].append({
                            "Model": model_name,
                            **{feature: input_values[i][j] for j, feature in enumerate(feature_list)},
                            "Prediction": prediction
                        })
                except Exception as e:
                    predictions.append({"Model": model_name, "Prediction": f"Lỗi: {str(e)}"})
            else:
                predictions.append({"Model": model_name, "Prediction": "Không tìm thấy mô hình."})

        # Hiển thị kết quả
        st.write("Kết quả dự đoán từ tất cả các mô hình:")
        predictions_df = pd.DataFrame(predictions)
        st.dataframe(predictions_df)

# Hiển thị lịch sử dự đoán
st.header("Lịch sử dự đoán")
if len(st.session_state["history"]) > 0:
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df)

    # Nút tải về file CSV
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Tải lịch sử dự đoán",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.write("Chưa có lịch sử dự đoán.")

# Trực quan hóa dự đoán so với thực tế
st.header("Dự đoán so với Thực tế")
if input_mode == "Dự đoán từ bộ dữ liệu ban đầu" and prediction_mode == "Dự đoán bằng tất cả các mô hình":
    st.write("So sánh giữa các mô hình:")
    comparison_results = []
    for model_name, model_file in zip(model_names, model_files):
        model_file_path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(model_file_path):
            model = joblib.load(model_file_path)
            try:
                y_pred = model.predict(input_values)
                comparison_results.append({
                    "Model": model_name,
                    "Actual": selected_y_actual.values.flatten(),
                    "Predicted": y_pred.flatten()
                })
            except Exception as e:
                st.error(f"Lỗi khi dự đoán với {model_name}: {e}")
    for result in comparison_results:
        fig = px.scatter(x=result["Actual"], y=result["Predicted"],
                         labels={"x": "Thực tế", "y": "Dự đoán"},
                         title=f"So sánh Dự đoán vs Thực tế ({result['Model']})")
        fig.add_shape(
            type="line", x0=min(result["Actual"]), y0=min(result["Actual"]),
            x1=max(result["Actual"]), y1=max(result["Actual"]),
            line=dict(color="Red", dash="dash")
        )
        st.plotly_chart(fig)

# Trực quan hóa SHAP và PDP
st.header("Trực quan hóa SHAP và PDP")

# Hiển thị các biểu đồ SHAP
st.sidebar.header("Trực quan hóa SHAP")
shap_model = st.sidebar.selectbox("Chọn mô hình để xem SHAP:", options=model_names)
shap_feature = st.sidebar.selectbox("Chọn đặc trưng để xem SHAP:", options=feature_list)
shap_file = os.path.join(DATA_DIR, f"{shap_model.replace(' ', '_')}_shap_values.npy")

if os.path.exists(shap_file):
    shap_values = np.load(shap_file)
    feature_index = feature_list.index(shap_feature)
    shap_feature_values = shap_values[:, feature_index]

    st.subheader(f"Giá trị SHAP cho đặc trưng: {shap_feature} - Mô hình: {shap_model}")
    st.bar_chart(pd.DataFrame(shap_feature_values, columns=["SHAP Value"]))
else:
    st.warning("Không tìm thấy giá trị SHAP. Vui lòng kiểm tra lại.")

# Trực quan hóa PDP
st.sidebar.header("Trực quan hóa PDP")
pdp_model = st.sidebar.selectbox("Chọn mô hình để xem PDP:", options=model_names)
pdp_feature = st.sidebar.selectbox("Chọn đặc trưng để vẽ PDP:", options=feature_list)
pdp_path = os.path.join(DATA_DIR, f"pdp_{pdp_model.replace(' ', '_')}_{pdp_feature}.png")

if os.path.exists(pdp_path):
    st.subheader(f"Partial Dependence Plot cho đặc trưng: {pdp_feature} - Mô hình: {pdp_model}")
    st.image(pdp_path, caption=f"PDP cho {pdp_feature} - Mô hình: {pdp_model}")
else:
    st.warning("Không tìm thấy biểu đồ PDP. Vui lòng kiểm tra lại.")
