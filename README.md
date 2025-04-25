# GPT-4 Facial Emotion Recognition Analysis
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This project is part of an **Honors Thesis** and serves as a **technical evaluation of GPT-4’s vision capabilities** to recognize and reason about human emotion from facial expressions. It connects GPT-4’s responses to established psychological theories on emotion recognition, particularly drawing from the work of **Paul Ekman** and his research on basic emotional expressions.

---

## Objective

To determine whether GPT-4 can:

- Accurately identify individual facial emotions.
- Estimate average group emotions from matrixed faces.
- Match or exceed human-level performance benchmarks.
- Reveal performance differences across orientation (upright vs. inverted) and presentation type (single vs. matrixed).

---

## License
This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this software for academic or commercial purposes, as long as you include the original copyright.

If you use this project in academic work, please cite the original thesis:

Shafiei, G. (2025). Emotion recognition through GPT-4 computer vision analysis of facial expressions [Honors thesis, University of North Carolina at Charlotte]. GitHub. https://github.com/GreysonShafiei/GPT-4_Facial_Expressions

---

## Project Structure

```bash
GPT-4_Facial_Expressions/
├── .env                                  # Contains your OpenAI API key
├── .gitignore
├── README.md                             # This file
│
├── CSV_AnswerKey_Cleaner.py              # Cleans and converts the WSEFEP emotion labels
├── GPT-4_Image_Processing.py             # Submits images to GPT-4 and stores responses
├── ImageRotator.py                       # Rotates upright images 180° to create inverted sets
├── WSEFEP - norms & FACS - Arkusz1.csv   # Original label data from the WSEFEP dataset
│
├── Images/                               # Organized facial expression images
│   ├── Matrix_inverted/
│   ├── Matrix_upright/
│   ├── Single_inverted/
│   └── Single_upright/
│
├── Results/                              # GPT-4 responses, analysis scripts, and outputs
│   ├── ResultsAnalysis.py                # Runs statistical tests and generates plots
│   ├── MatrixFace_answers.csv            # Ground truth for matrixed face images
│   ├── SingleFace_answers.csv            # Ground truth for single face images
│   ├── gpt4_image_responses_matrix_inverted_matrix_exp.csv
│   ├── gpt4_image_responses_matrix_inverted_matrix_noExp.csv
│   ├── gpt4_image_responses_matrix_upright_matrix_exp.csv
│   ├── gpt4_image_responses_matrix_upright_matrix_noExp.csv
│   ├── gpt4_image_responses_single_inverted_single_exp.csv
│   ├── gpt4_image_responses_single_inverted_single_noExp.csv
│   ├── gpt4_image_responses_single_upright_single_exp.csv
│   ├── gpt4_image_responses_single_upright_single_noExp.csv
│   └── FinalResults/
│       ├── ML_Analysis_GPT4.py
│       ├── final_results.csv
│       ├── hypothesis_summary.csv
│       ├── GPT4_Analysis_Report.pdf
│       ├── GPT4_ML_Analysis_Report.pdf
│       ├── accuracy_by_face_type.png
│       ├── accuracy_by_orientation.png
│       ├── accuracy_by_prompt_style.png
│       ├── accuracy_by_emotion_and_face_type.png
│       ├── accuracy_by_emotion_and_orientation.png
│       ├── conf_matrix_Overall.png
│       ├── conf_matrix_Upright.png
│       ├── conf_matrix_Inverted.png
│       ├── ML_classification_report.png
│       ├── ML_confusion_matrix.png
│       ├── ML_feature_importances.png
│       ├── ML_features_scaled.png
│       └── ML_kmeans_clusters.png
├── OldTestPromptResults/                 # Legacy CSV files from earlier test runs
```

## Required Libraries for Reproduction

To successfully reproduce this study, you must:

- Obtain an OpenAI API key and store it in a `.env` file.
- Install the following Python libraries in your environment:

| Library         | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `openai`        | Access to GPT-4 Vision model via OpenAI API                             |
| `python-dotenv` | Load the `.env` file that contains your OpenAI API key                  |
| `pandas`        | Data manipulation and merging GPT results with answer keys              |
| `matplotlib`    | Visualization of accuracy and ML features                               |
| `seaborn`       | Statistical visualizations with confidence intervals                    |
| `scipy`         | Statistical testing (t-tests, z-tests)                                  |
| `statsmodels`   | Proportions z-test and statistical modeling support                     |
| `fpdf`          | PDF generation for final and ML analysis reports                        |
| `Pillow`        | Automated image rotation for inverted face testing                      |
| `scikit-learn`  | Machine learning (Random Forest, PCA, clustering, confusion matrices)   |

To install everything at once:

```bash
pip install openai python-dotenv pandas matplotlib seaborn scipy statsmodels fpdf Pillow scikit-learn
```