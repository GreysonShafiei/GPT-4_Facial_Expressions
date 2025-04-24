import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages

# Load data
df = pd.read_csv('final_results.csv')

# PDF output
pdf = PdfPages('GPT4_ML_Analysis_Report.pdf')

# ------ Predictive analysis of the final_results ------

# Encode categorical variables
categorical_vars = ['orientation', 'face_type', 'prompt_style', 'correct_emotion']
df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Correct columns removal
X = df_encoded.drop(['correct', 'response', 'image_name_x', 'image_name_y', 'img_key', 'full_response'], axis=1)
y = df_encoded['correct'].astype(int)

# Split dataset (70% 30% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap='Blues')
plt.title('Confusion Matrix', pad=20)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar(cax)
plt.grid(False)
plt.tight_layout()
pdf.savefig(fig)
plt.savefig('ML_confusion_matrix.png')
plt.close()

# Feature importance plot
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
feature_importances.tail(10).plot(kind='barh', ax=ax)
ax.set_title('Top 10 Feature Importances')
plt.tight_layout()
pdf.savefig(fig)
plt.savefig('ML_feature_importances.png')
plt.close()

# Add accuracy and export to the PDF
fig, ax = plt.subplots(figsize=(8, 6))
plt.axis('off')
plt.title('Classification Report', fontsize=14)
report_text = f"Accuracy: {acc:.2f}\n\nClassification Report:\n{report}"
plt.text(0, 1, report_text, fontsize=10, va='top', ha='left', family='monospace')
pdf.savefig(fig)
plt.savefig('ML_classification_report.png')
plt.close()

# ------ Exploratory analysis of the final_results ------

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

# PCA plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=df['correct'], cmap='viridis', alpha=0.7)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('PCA of GPT-4 Emotion Recognition')
plt.colorbar(scatter, label='Correct (1) vs Incorrect (0)')
plt.grid(True)
plt.tight_layout()
pdf.savefig(fig)
plt.savefig('ML_features_scaled.png')
plt.close()

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Clustering plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='Set1', alpha=0.7)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('K-Means Clustering of GPT-4 Emotion Recognition')
legend1 = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend1)
plt.grid(True)
plt.tight_layout()
pdf.savefig(fig)
plt.savefig('ML_kmeans_clusters.png')
plt.close()

pdf.close()