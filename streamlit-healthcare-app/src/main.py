import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import boto3
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time as timer
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pickle  # Import the pickle library

# aws_access_key = 'AKIAYWEQCPSZKVPCUMOF'
# aws_secret_key = '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'
# region = 'ap-southeast-2'
# bucket_name = 'my-capstone-bucket1'
# key = 'Bucket/processed_healthcare_data.csv'
local_filename = 'processed_healthcare_data.csv'

# try:
#     s3 = boto3.client('s3',
#                       aws_access_key_id=aws_access_key,
#                       aws_secret_access_key=aws_secret_key,
#                       region_name=region)
#
#     s3.download_file(Bucket=bucket_name, Key=key, Filename=local_filename)
#     print(f" Đã tải {key} về dưới tên {local_filename}")
# except Exception as e:
#     print(f"Lỗi khi tải file từ S3: {e}. Đảm bảo file '{local_filename}' đã có sẵn nếu bạn chạy cục bộ.")

data = pd.read_csv(local_filename)

# Tách X, y
X = data.loc[:, data.columns != 'stroke']
y = data['stroke']

# Xử lý imbalance bằng SMOTE vì dữ liệu lưu trên s3 chỉ tiền xử lý cơ bản chứ ko lưu smote
smote = SMOTE(sampling_strategy='minority', random_state=42)
X, y = smote.fit_resample(X, y)
print(f" Dữ liệu sau SMOTE: X = {X.shape}, y = {y.shape}")
unique_classes_smote, counts_smote = np.unique(y, return_counts=True)
print("  Số lượng mẫu của từng lớp sau SMOTE:")
for cls, count in zip(unique_classes_smote, counts_smote):
    print(f"    Lớp {cls}: {count} mẫu")

def split_train_valid_test(X, y, test_size=0.1, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size / (1 - test_size),
                                                          random_state=random_state, stratify=y_train)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(X, y, test_size=0.1, random_state=42)
_, train_counts = np.unique(y_train, return_counts=True)
_, valid_counts = np.unique(y_valid, return_counts=True)
_, test_counts = np.unique(y_test, return_counts=True)
print("[train] # class 0: {} | # class 1: {}".format(train_counts[0], train_counts[1]))
print("[valid] # class 0: {} | # class 1: {}".format(valid_counts[0], valid_counts[1]))
print("[test]  # class 0: {} | # class 1: {}".format(test_counts[0], test_counts[1]))

# Data Normalisation
scaler = StandardScaler()
scaler = scaler.fit(X_train)  # Fit scaler chỉ trên tập huấn luyện

X_train_std = scaler.transform(X_train)
X_valid_std = scaler.transform(X_valid)
X_test_std = scaler.transform(X_test)


# Hàm tính Sensitivity và Specificity (giữ nguyên)
def calc_sens_spec(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    # calculate the sensitivity
    sensitivity = TP / (TP + FN)
    # calculate the specificity
    specificity = TN / (TN + FP)
    return sensitivity, specificity


# --- Huấn luyện và đánh giá Random Forest (mô hình bạn chọn) ---
print("\n--- Huấn luyện và đánh giá mô hình Random Forest ---")
start = timer.time()
ranfor_model = RandomForestClassifier(n_estimators=100, random_state=42)
ranfor_model.fit(X_train_std, y_train)
end = timer.time()
print("Finished training Random Forest within {:.2f} seconds".format(end - start))

# Predicting the test set results
y_ranfor = ranfor_model.predict(X_test_std)
y_ranfor_prob = ranfor_model.predict_proba(X_test_std)

print("Classification report for Random Forest: \n{}".format(classification_report(y_test, y_ranfor)))
print("Confusion matrix for Random Forest: \n{}".format(confusion_matrix(y_test, y_ranfor)))
print("Accuracy score for Random Forest: {:.2f}".format(accuracy_score(y_test, y_ranfor)))

# Calculate precision, recall, and f1 scores
prec_ranfor = precision_score(y_test, y_ranfor)
rec_ranfor = recall_score(y_test, y_ranfor)
f1_ranfor = f1_score(y_test, y_ranfor)
print("Precision score for Random Forest: {:.2f}".format(prec_ranfor))
print("Recall score for Random Forest: {:.2f}".format(rec_ranfor))
print("F1 score for Random Forest: {:.2f}".format(f1_ranfor))

# Calculate sensitivity, specificity, and auc
sens_ranfor, spec_ranfor = calc_sens_spec(y_test, y_ranfor)
fpr, tpr, _ = roc_curve(y_test, y_ranfor_prob[:, 1])
auc_ranfor = roc_auc_score(y_test, y_ranfor_prob[:, 1])
print("Sensitivity score for Random Forest: {:.2f}".format(sens_ranfor))
print("Specitivity score for Random Forest: {:.2f}".format(spec_ranfor))
print("AUC score for Random Forest: {:.2f}".format(auc_ranfor))

# Plot ROC curve for Random Forest
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_ranfor)
ax.plot([0, 1], [0, 1], color='green', linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (Random Forest)')
ax.legend(loc="lower right")
plt.show()

# --- Đóng gói mô hình và StandardScaler bằng pickle ---
print("\n--- Đóng gói mô hình Random Forest và Scaler ---")

# Đóng gói mô hình Random Forest
final_model = ranfor_model
try:
    with open('./random_forest_model.sav', 'wb') as file:
        pickle.dump(final_model, file)
    print("Mô hình Random Forest đã được đóng gói và lưu vào 'random_forest_model.sav'")
except Exception as e:
    print(f"Lỗi khi đóng gói mô hình: {e}")

# Cũng quan trọng là đóng gói StandardScaler
# vì sẽ cần nó để chuẩn hóa dữ liệu mới trước khi đưa vào mô hình đã huấn luyện
try:
    with open('./scaler.sav', 'wb') as file:
        pickle.dump(scaler, file)
    print("Scaler đã được đóng gói và lưu vào 'scaler.sav'")
except Exception as e:
    print(f"Lỗi khi đóng gói scaler: {e}")
