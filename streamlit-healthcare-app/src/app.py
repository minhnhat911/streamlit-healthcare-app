import streamlit as st
import pickle
import numpy as np

# Tải mô hình đã huấn luyện và bộ chuẩn hóa
import os

# Lấy đường dẫn thư mục hiện tại của file app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Tạo đường dẫn đầy đủ đến file mô hình
model_path = os.path.join(current_dir, 'random_forest_model.sav')

# Load mô hình
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
# Tạo đường dẫn đến file scaler
scaler_path = os.path.join(current_dir, 'scaler.sav')

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    
# Hàm để dự đoán đột quỵ
def predict_stroke(features):
    features = np.array(features).reshape(1, -1)
    features_std = scaler.transform(features)
    prediction = model.predict(features_std)
    probability = model.predict_proba(features_std)[0][1]
    return prediction, probability

# Giao diện người dùng Streamlit
def main():
    # Tùy chỉnh tiêu đề ứng dụng với HTML/CSS
    st.markdown("""
    <div style="background-color: #1E3A8A; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center; font-size: 42px; font-weight: bold; text-shadow: 2px 2px 4px #000000;">
            🏥 ỨNG DỤNG DỰ ĐOÁN ĐỘT QUỴ
        </h1>

    </div>
    """, unsafe_allow_html=True)

    # Chia giao diện thành 2 cột
    col1, col2 = st.columns(2)
    
    # Cột thông tin cá nhân
    with col1:
        st.subheader("Thông tin cá nhân")
        age = st.number_input("Tuổi", min_value=1, max_value=100, value=30)
        gender = st.selectbox("Giới tính", ("Nam", "Nữ"))
        ever_married = st.selectbox("Đã kết hôn", ("Có", "Không"))
        work_type = st.selectbox("Loại hình công việc", ("Công ty tư nhân", "Làm nghề tự do", "Trẻ em", "Công chức", "Chưa làm việc"))
        residence_type = st.selectbox("Loại hình cư trú", ("Thành thị", "Nông thôn"))

    # Cột thông tin sức khỏe
    with col2:
        st.subheader("Thông tin sức khỏe")
        hypertension = st.selectbox("Tăng huyết áp", ("Có", "Không"))
        heart_disease = st.selectbox("Bệnh tim", ("Có", "Không"))
        avg_glucose_level = st.number_input("Mức đường huyết trung bình", min_value=0.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, value=20.0)
        smoking_status = st.selectbox("Tình trạng hút thuốc", ("Không rõ", "Đã từng hút thuốc", "Chưa bao giờ hút thuốc", "Đang hút thuốc"))

    # Chuyển đổi gia trị đầu vào thành định dạng số
    hypertension = 1 if hypertension == "Có" else 0
    heart_disease = 1 if heart_disease == "Có" else 0
    gender = 1 if gender == "Nam" else 0
    ever_married = 1 if ever_married == "Có" else 0
    residence_type = 1 if residence_type == "Thành thị" else 0

    smoking_map = {
        "Không rõ": 0,
        "Đã từng hút thuốc": 1,
        "Chưa bao giờ hút thuốc": 2,
        "Đang hút thuốc": 3
    }
    smoking_status = smoking_map[smoking_status]

    work_type_map = {
        "Công chức": 0,
        "Chưa làm việc": 1,
        "Công ty tư nhân": 2,
        "Làm nghề tự do": 3,
        "Trẻ em": 4,
    }
    work_type = work_type_map[work_type]

    # Nút dự đoán ở giữa dưới 2 cột
    if st.button("Dự đoán đột quỵ", type="primary"):
        # Hiển thị spinner khi đang xử lý
        with st.spinner("Đang xử lý dự đoán..."):
            features = [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]
            prediction, probability = predict_stroke(features)
            
            # Hiển thị kết quả dự đoán
            st.markdown("""
            <div style="background-color: #F9FAFB; border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="color: #1E3A8A; text-align: center; margin-bottom: 20px; font-size: 28px; border-bottom: 2px solid #E5E7EB; padding-bottom: 10px;">
                    📊 KẾT QUẢ DỰ ĐOÁN
                </h2>
            """, unsafe_allow_html=True)
            
            if prediction[0] == 0:
                # Kết quả nguy cơ thấp - màu xanh
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 25px;">
                    <div style="background-color: #DCFCE7; width: 80%; border-radius: 10px; padding: 20px; border-left: 8px solid #059669; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                        <h3 style="color: #059669; text-align: center; font-size: 24px; margin-bottom: 15px;">
                            ✅ NGUY CƠ ĐỘT QUỴ THẤP
                        </h3>
                        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                            <div style="font-size: 18px; font-weight: bold; color: #374151;">Dự đoán:</div>
                            <div style="font-size: 18px; font-weight: bold; color: #059669;">Nguy cơ thấp</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                            <div style="font-size: 18px; font-weight: bold; color: #374151;">Độ tin cậy:</div>
                            <div style="font-size: 18px; font-weight: bold; color: #059669;">{(1-probability)*100:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Phần gợi ý cho nguy cơ thấp
                st.markdown("""
                <div style="background-color: #EFF6FF; border-radius: 10px; padding: 15px; margin: 10px 0 20px 0; border-left: 5px solid #3B82F6;">
                    <h4 style="color: #1E40AF; margin-bottom: 10px; font-size: 18px;">💡 Gợi ý:</h4>
                    <ul style="margin-left: 20px; color: #1F2937;">
                        <li>Tiếp tục duy trì lối sống lành mạnh</li>
                        <li>Tập thể dục thường xuyên</li>
                        <li>Kiểm tra sức khỏe định kỳ</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                # Kết quả nguy cơ cao - màu đỏ
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 25px;">
                    <div style="background-color: #FEE2E2; width: 80%; border-radius: 10px; padding: 20px; border-left: 8px solid #DC2626; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                        <h3 style="color: #DC2626; text-align: center; font-size: 24px; margin-bottom: 15px;">
                            ⚠️ NGUY CƠ ĐỘT QUỴ CAO
                        </h3>
                        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                            <div style="font-size: 18px; font-weight: bold; color: #374151;">Dự đoán:</div>
                            <div style="font-size: 18px; font-weight: bold; color: #DC2626;">Nguy cơ cao</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                            <div style="font-size: 18px; font-weight: bold; color: #374151;">Độ tin cậy:</div>
                            <div style="font-size: 18px; font-weight: bold; color: #DC2626;">{probability*100:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Phần cảnh báo và khuyến nghị cho nguy cơ cao
                st.markdown("""
                <div style="background-color: #FEF3C7; border-radius: 10px; padding: 15px; margin: 10px 0 20px 0; border-left: 5px solid #F59E0B;">
                    <h4 style="color: #B45309; margin-bottom: 10px; font-size: 18px;">⚠️ Khuyến nghị quan trọng:</h4>
                    <ul style="margin-left: 20px; color: #1F2937;">
                        <li>Gặp bác sĩ càng sớm càng tốt</li>
                        <li>Kiểm soát các yếu tố nguy cơ</li>
                        <li>Theo dõi huyết áp và đường huyết thường xuyên</li>
                        <li>Thay đổi chế độ ăn uống và sinh hoạt</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
                          

#Hàm main
if __name__ == "__main__":
    main()
