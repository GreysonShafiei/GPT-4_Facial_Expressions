import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fpdf import FPDF
from statsmodels.stats.proportion import proportions_ztest

# Constants
RESULTS_FOLDER = "../Results"
SINGLE_CSV = "SingleFace_answers.csv"
MATRIX_CSV = "MatrixFace_answers.csv"
CHANCE_LEVEL = 1 / 6  # Random Chance Statistic For Comparison
FINAL_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "FinalResults")
os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)

# Load and normalize answers
single_answers = pd.read_csv(os.path.join(RESULTS_FOLDER, SINGLE_CSV))
matrix_answers = pd.read_csv(os.path.join(RESULTS_FOLDER, MATRIX_CSV))
single_answers["correct_emotion"] = single_answers["correct_emotion"].str.strip().str.lower()
matrix_answers["correct_emotion"] = matrix_answers["correct_emotion"].str.strip().str.lower()

# Merge GPT results with correct answers
all_data = []
for filename in os.listdir(RESULTS_FOLDER):
    if filename.startswith("gpt4_image_responses") and filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(RESULTS_FOLDER, filename))
        df["image_name"] = df["image_name"].str.strip()
        df["response"] = df["response"].str.strip().str.lower()

        if "single" in filename.lower():
            answer_df = single_answers.copy()
            df["face_type"] = "individual"
        elif "matrix" in filename.lower():
            answer_df = matrix_answers.copy()
            df["face_type"] = "matrixed"
        else:
            continue

        df["orientation"] = "upright" if "upright" in filename.lower() else "inverted"
        merged = pd.merge(df, answer_df, on="image_name", how="inner")
        merged["correct"] = merged["response"] == merged["correct_emotion"]
        all_data.append(merged)

# Combine all merged data
full_df = pd.concat(all_data, ignore_index=True)
full_df.to_csv(os.path.join(FINAL_RESULTS_PATH, "final_results.csv"), index=False)

# Hypothesis Tests
results = []

# H1: One-sample t-test vs. 80%
t1, p1 = stats.ttest_1samp(full_df["correct"], popmean=0.80)
results.append(("H1", "One-sample t-test (vs 80%)", t1, p1))

# H2: Paired t-test (upright vs inverted)
upright = full_df[full_df["orientation"] == "upright"]["correct"]
inverted = full_df[full_df["orientation"] == "inverted"]["correct"]
min_len = min(len(upright), len(inverted))
t2, p2 = stats.ttest_rel(upright.iloc[:min_len], inverted.iloc[:min_len])
results.append(("H2", "Paired t-test (upright vs inverted)", t2, p2))

# H3: Independent t-test (individual vs matrixed)
indiv = full_df[full_df["face_type"] == "individual"]["correct"]
matrix = full_df[full_df["face_type"] == "matrixed"]["correct"]
t3, p3 = stats.ttest_ind(indiv, matrix)
results.append(("H3", "Independent t-test (individual vs matrixed)", t3, p3))

# H4: Chi-square test vs. chance for matrixed
matrix_only = full_df[full_df["face_type"] == "matrixed"]
correct_n = matrix_only["correct"].sum()
total_n = len(matrix_only)
z4, p4 = proportions_ztest(count=correct_n, nobs=total_n, value=CHANCE_LEVEL)
results.append(("H4", "Chi-square test (matrix vs chance)", z4, p4))

# Summary Table
summary_df = pd.DataFrame(results, columns=["Hypothesis", "Test", "Stat", "P-value"])
summary_df["Significant"] = summary_df["P-value"] < 0.05
summary_df.to_csv(os.path.join(FINAL_RESULTS_PATH, "hypothesis_summary.csv"), index=False)

# Visualization

# Accuracy by face type
plt.figure(figsize=(6,4))
sns.barplot(data=full_df, x="face_type", y="correct", estimator=lambda x: sum(x)/len(x), ci=95)
plt.title("Accuracy by Face Type")
plt.ylabel("Accuracy")
plt.xlabel("Face Type")
plt.ylim(0,1)
face_plot_path = os.path.join(FINAL_RESULTS_PATH, "accuracy_by_face_type.png")
plt.savefig(face_plot_path)
plt.close()

# Accuracy by orientation
plt.figure(figsize=(6,4))
sns.barplot(data=full_df, x="orientation", y="correct", estimator=lambda x: sum(x)/len(x), ci=95)
plt.title("Accuracy by Orientation")
plt.ylabel("Accuracy")
plt.xlabel("Orientation")
plt.ylim(0,1)
orient_plot_path = os.path.join(FINAL_RESULTS_PATH, "accuracy_by_orientation.png")
plt.savefig(orient_plot_path)
plt.close()

# PDF of Results

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "GPT-4 Facial Emotion Analysis", ln=True, align="C")
        self.ln(10)

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(4)

    def section_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, text)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.section_title("Summary of Hypothesis Tests")
for _, row in summary_df.iterrows():
    text = f"{row['Hypothesis']} - {row['Test']}\nStat: {row['Stat']:.3f}, P-value: {row['P-value']:.4f} → {'Significant' if row['Significant'] else 'Not Significant'}"
    pdf.section_body(text)

pdf.section_title("Visualizations")

pdf.image(face_plot_path, w=160)
pdf.ln(10)
pdf.image(orient_plot_path, w=160)

pdf_path = os.path.join(FINAL_RESULTS_PATH, "GPT4_Analysis_Report.pdf")
pdf.output(pdf_path)

print(f"\nReport saved to: {pdf_path}")