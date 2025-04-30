def main():
    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from fpdf import FPDF
    from statsmodels.stats.proportion import proportions_ztest
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    # Constants
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    os.chdir(PROJECT_ROOT)
    CHANCE_LEVEL = 1 / 7  # random‑chance baseline for 7 emotions
    RESULTS_PATH = os.path.join(PROJECT_ROOT, "Results")
    FINAL_RESULTS_PATH = os.path.join(RESULTS_PATH, "FinalResults")
    print("\nFiles in 'Results/' folder:")
    os.listdir(".")
    os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)

    # Load answer keys
    single_answers = pd.read_csv(os.path.join(RESULTS_PATH, "SingleFace_answers.csv"))
    matrix_answers = pd.read_csv(os.path.join(RESULTS_PATH, "MatrixFace_answers.csv"))
    single_answers["correct_emotion"] = single_answers["correct_emotion"].str.strip().str.lower()
    matrix_answers["correct_emotion"] = matrix_answers["correct_emotion"].str.strip().str.lower()

    # Merge GPT results with correct answers
    all_data = []
    file_pat = re.compile(r"gpt4_image_responses_(.+?)_(matrix|single)_(exp|noexp)\.csv", re.I)
    for fname in os.listdir(RESULTS_PATH):
        m = file_pat.match(fname)
        if not m:
            continue

        # Checks csv for the various features of category, face type, and explanation
        category_name, face_token, exp_token = m.groups()
        prompt_style = "explanation" if exp_token.lower() == "exp" else "no_explanation"
        face_type = "matrixed" if face_token.lower() == "matrix" else "individual"
        orientation = "upright" if "upright" in category_name.lower() else "inverted"

        df = pd.read_csv(os.path.join(RESULTS_PATH, fname))
        df["image_name"] = df["image_name"].str.strip().str.lower()
        df["response"] = df["response"].str.strip().str.lower()
        df["face_type"] = face_type
        df["orientation"] = orientation
        df["prompt_style"] = prompt_style

        answer_df = single_answers.copy() if face_type == "individual" else matrix_answers.copy()
        answer_df["image_name"] = answer_df["image_name"].str.strip().str.lower()

        df["img_key"] = df["image_name"].str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
        answer_df["img_key"] = answer_df["image_name"].str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

        merged = pd.merge(df, answer_df, on="img_key", how="inner")
        merged["correct"] = merged["response"] == merged["correct_emotion"]
        all_data.append(merged)

    if not all_data:
        print("No GPT‑4 CSVs found.")
        return

    # Save the compiled final results
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(os.path.join(FINAL_RESULTS_PATH, "final_results.csv"), index=False)

    print("\nSample counts by face type:\n", full_df["face_type"].value_counts())
    print("\nSample counts by orientation:\n", full_df["orientation"].value_counts())

    # Descriptive statistics
    # 1 = correct, 0 = incorrect
    full_df["correct_numeric"] = full_df["correct"].astype(int)

    face_map = {"individual": "Individual", "matrixed": "Matrixed"}
    orient_map = {"upright": "Upright", "inverted": "Inverted"}
    prompt_map = {"no_explanation": "No explanation", "explanation": "Explanation"}

    full_df["Face type"] = full_df["face_type"].map(face_map)
    full_df["Orientation"] = full_df["orientation"].map(orient_map)
    full_df["Prompt style"] = full_df["prompt_style"].map(prompt_map)

    desc_df = (full_df.groupby(["Face type", "Orientation", "Prompt style"])["correct_numeric"].agg(N="size", Accuracy_Mean="mean", Accuracy_SD=lambda x: x.std(ddof=1)).reset_index().sort_values(["Face type", "Orientation", "Prompt style"]))

    # Save the results into a csv
    desc_df[["Accuracy_Mean", "Accuracy_SD"]] = desc_df[["Accuracy_Mean", "Accuracy_SD"]].round(3)
    desc_stats_path = os.path.join(FINAL_RESULTS_PATH, "descriptive_statistics.csv")
    desc_df.to_csv(desc_stats_path, index=False)
    print(f"Descriptive statistics saved to: {desc_stats_path}")

    # Hypothesis tests
    results = []

    # H1: One-sample t-test vs. Human Average (About 82.35%)
    t1, p1 = stats.ttest_1samp(full_df["correct"], popmean=0.8234761904761905)
    results.append(("H1", "One-sample t-test vs human average", t1, p1))

    # H2: Paired t-test (upright vs inverted)
    upright = full_df[full_df["orientation"] == "upright"]["correct"].astype(int)
    inverted = full_df[full_df["orientation"] == "inverted"]["correct"].astype(int)
    min_len = min(len(upright), len(inverted))
    t2, p2 = stats.ttest_rel(upright.iloc[:min_len], inverted.iloc[:min_len])
    results.append(("H2", "Paired t-test upright vs inverted", t2, p2))

    # H3: Independent t-test (individual vs matrixed)
    img_level = (full_df.groupby(['img_key', 'face_type'], as_index=False)['correct'].mean())
    indiv = img_level[img_level['face_type'] == 'individual']['correct']
    matrix = img_level[img_level['face_type'] == 'matrixed']['correct']
    t3, p3 = stats.ttest_ind(indiv, matrix, equal_var=False)
    results.append(("H3", "Welch t-test individual vs matrixed", t3, p3))

    # H4: Chi-square test vs. chance for matrixed
    matrix_only = full_df[full_df["face_type"] == "matrixed"]
    z4, p4 = proportions_ztest(count=matrix_only["correct"].sum(), nobs=len(matrix_only), value=CHANCE_LEVEL)
    results.append(("H4", "Proportion z-test matrix vs chance", z4, p4))

    # H5: paired t‑test (explanation vs no‑explanation)
    exp = full_df[full_df["prompt_style"] == "explanation"]["correct"].astype(int)
    noexp = full_df[full_df["prompt_style"] == "no_explanation"]["correct"].astype(int)
    min_len = min(len(exp), len(noexp))
    t5, p5 = stats.ttest_rel(exp.iloc[:min_len], noexp.iloc[:min_len])
    results.append(("H5", "Paired t-test explanation vs no-explanation", t5, p5))

    summary_df = (pd.DataFrame(results, columns=["Hypothesis", "Test", "Stat", "P-value"]).assign(Significant=lambda d: d["P-value"] < 0.05))
    summary_df.to_csv(os.path.join(FINAL_RESULTS_PATH, "hypothesis_summary.csv"), index=False)

    # Visualizations
    def _bar(col, xlab, title, fname):
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(data=full_df, x=col, y="correct", estimator=lambda x: sum(x) / len(x), errorbar=("ci", 95))
        ax.set_xlabel(xlab)
        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        path = os.path.join(FINAL_RESULTS_PATH, fname)
        plt.savefig(path)
        plt.close()
        return path

    face_plot_path = _bar("face_type", "Face Type", "Accuracy by Face Type", "accuracy_by_face_type.png")
    orient_plot_path = _bar("orientation", "Orientation", "Accuracy by Orientation", "accuracy_by_orientation.png")
    prompt_plot = _bar("prompt_style", "Prompt Style", "Accuracy by Prompt Style", "accuracy_by_prompt_style.png")

    emotion_face = (full_df.assign(is_correct=lambda d: d["correct"].astype(int)).groupby(["face_type", "correct_emotion"], as_index=False).agg(accuracy=("is_correct", "mean")))
    plt.figure(figsize=(10, 5))
    sns.barplot(data=emotion_face, x="correct_emotion", y="accuracy", hue="face_type", errorbar=("ci", 95))
    plt.title("Per-Emotion Accuracy by Face Type")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.legend(title="Face Type", loc="upper right")
    face_em_plot = os.path.join(FINAL_RESULTS_PATH, "accuracy_by_emotion_and_face_type.png")
    plt.savefig(face_em_plot)
    plt.close()

    emotion_orient = (full_df.assign(is_correct=lambda d: d["correct"].astype(int)).groupby(["orientation", "correct_emotion"], as_index=False).agg(accuracy=("is_correct", "mean")))
    plt.figure(figsize=(10, 5))
    sns.barplot(data=emotion_orient, x="correct_emotion", y="accuracy", hue="orientation", errorbar=("ci", 95))
    plt.title("Per-Emotion Accuracy by Orientation")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.legend(title="Orientation", loc="upper right")
    orient_em_plot = os.path.join(FINAL_RESULTS_PATH, "accuracy_by_emotion_and_orientation.png")
    plt.savefig(orient_em_plot)
    plt.close()

    # Confusion matrix
    def plot_conf_matrix(df, group_label):
        y_true = df["correct_emotion"]
        y_pred = df["response"]
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=45)
        ax.set_title(f'Confusion Matrix: {group_label}')
        plt.tight_layout()
        path = os.path.join(FINAL_RESULTS_PATH, f'conf_matrix_{group_label}.png')
        plt.savefig(path)
        plt.close()
        return path

    conf_matrix_overall_path = plot_conf_matrix(full_df, "Overall")
    conf_matrix_upright = plot_conf_matrix(full_df[full_df["orientation"] == "upright"], "Upright")
    conf_matrix_inverted = plot_conf_matrix(full_df[full_df["orientation"] == "inverted"], "Inverted")

    # PDF report
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

    # Summarize the calculations results
    pdf.section_title("Summary of Hypothesis Tests")
    for _, row in summary_df.iterrows():
        sig = "Significant" if row["Significant"] else "Not Significant"
        txt = (
            f"{row['Hypothesis']} - {row['Test']}\n"
            f"Stat: {row['Stat']:.3f}, P-value: {row['P-value']:.4f} --> {sig}"
        )
        pdf.section_body(txt)

    pdf.add_page()

    # Add each of the plots
    pdf.section_title("Visualizations")
    for img in [prompt_plot, face_plot_path, orient_plot_path,
                face_em_plot, orient_em_plot, conf_matrix_overall_path, conf_matrix_upright, conf_matrix_inverted]:
        pdf.image(img, w=160)
        pdf.ln(6)

    pdf_path = os.path.join(FINAL_RESULTS_PATH, "GPT4_Analysis_Report.pdf")
    pdf.output(pdf_path)
    print(f"\nReport saved to: {pdf_path}")


if __name__ == "__main__":
    main()