import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import random

os.makedirs('image', exist_ok=True)

df = pd.read_csv('archive/framingham.csv')

df = df.dropna()

X = df.drop('TenYearCHD', axis=1).values
y = df['TenYearCHD'].values
feature_names = df.drop('TenYearCHD', axis=1).columns.tolist()

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Objective Function
def fitness_function(select_features_mask):
    if np.sum(select_features_mask) == 0:
        return 0.0
    
    X_train_sub = X_train[:, select_features_mask]
    X_test_sub = X_test[:, select_features_mask]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sub, y_train)
    return accuracy_score(y_test, knn.predict(X_test_sub))

def run_ewoa(num_features, num_whales=10, max_iter=20):
    whales = np.random.rand(num_whales, num_features)
    best_whale = None
    best_fitness = -1

    for iteration in range(max_iter):
        for i in range(num_whales):
            mask = whales[i] > 0.5 
            fitness = fitness_function(mask)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_whale = whales[i].copy()
                
        T = 2 - 2 * (iteration / max_iter)
        
        for i in range(num_whales):
            target_idx = random.randint(0, num_whales - 1)
            source_idx = random.randint(0, num_whales - 1)
            
            for j in range(num_features):
                r1, r2 = random.random(), random.random()
                A = 2 
                A1 = 2 * A * r1 - T
                C1 = 2 * r2
                
                D = abs(C1 * whales[target_idx, j] - whales[source_idx, j])
                new_position = whales[target_idx, j] - A1 * D
                
                whales[i, j] = np.clip(new_position, 0.0, 1.0)

    return best_whale > 0.5

optimal_features_mask = run_ewoa(num_features=X.shape[1], num_whales=10, max_iter=10)
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if optimal_features_mask[i]]

print(f"\nCác đặc trưng tối ưu được chọn: {selected_feature_names}")

X_train_opt = X_train[:, optimal_features_mask]
X_test_opt = X_test[:, optimal_features_mask]

print("\nĐang huấn luyện các mô hình phân loại trên tập đặc trưng đã rút gọn...")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(probability=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

hsvkn_model = VotingClassifier(
    estimators=[('svm', svm_model), ('knn', knn_model)],
    voting='soft'
)

hsvrf_model = VotingClassifier(
    estimators=[('svm', svm_model), ('rf', rf_model)],
    voting='soft'
)

models = {
    "LR": lr_model,
    "DT": dt_model,
    "KNN": knn_model,
    "SVM": svm_model,
    "RF": rf_model,
    "HSVKN": hsvkn_model,
    "HSVRF": hsvrf_model
}

results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}

for name, model in models.items():
    model.fit(X_train_opt, y_train)
    y_pred = model.predict(X_test_opt)
    
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["Precision"].append(prec)
    results["Recall"].append(rec)
    results["F1-Score"].append(f1)
    
    print(f"\n--- Mô hình: {name} ---")
    print(f"Độ chính xác (Accuracy): {acc:.2f}%")
    print(f"Độ chuẩn xác (Precision): {prec:.2f}%")
    print(f"Độ thu hồi (Recall): {rec:.2f}%")
    print(f"Điểm F1 (F1-Score): {f1:.2f}%")


x = np.arange(len(results["Model"]))  
width = 0.2  

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5*width, results["Accuracy"], width, label='Accuracy %')
rects2 = ax.bar(x - 0.5*width, results["Precision"], width, label='Precision %')
rects3 = ax.bar(x + 0.5*width, results["Recall"], width, label='Recall %')
rects4 = ax.bar(x + 1.5*width, results["F1-Score"], width, label='F1-Score %')

ax.set_ylabel('%')
ax.set_xlabel('Classification Methods')
ax.set_title('Performance Analysis of the Proposed System')
ax.set_xticks(x)
ax.set_xticklabels(results["Model"])
ax.set_ylim(0, 110)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)

plt.tight_layout()
plt.savefig('image/Figure_2_Performance_Analysis.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results["Model"], results["Accuracy"], marker='o', linestyle='-', linewidth=2, markersize=8)

ax.set_ylabel('%')
ax.set_xlabel('Classification Methods')
ax.set_title('Accuracy Analysis of EWOA')
ax.set_ylim(0, 110)

for i, txt in enumerate(results["Accuracy"]):
    ax.annotate(f"{txt:.2f}", (results["Model"][i], results["Accuracy"][i]), 
                textcoords="offset points", xytext=(0,10), ha='center')

plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('image/Figure_3_Accuracy_Analysis.png', dpi=300)
plt.close()

print("Hoàn tất")
