# GPT-4 Facial Emotion Recognition Analysis

This project is part of an **Honors Thesis** and serves as a **technical evaluation of GPT-4’s vision capabilities** to recognize and reason about human emotion from facial expressions. It connects GPT-4’s responses to established psychological theories on emotion recognition, particularly drawing from the work of **Paul Ekman** and his research on basic emotional expressions.

---

## Objective

To determine whether GPT-4 can:

- Accurately identify individual facial emotions.
- Estimate average group emotions from matrixed faces.
- Match or exceed human-level performance benchmarks.
- Reveal performance differences across orientation (upright vs. inverted) and presentation type (single vs. matrixed).

---

## Project Structure

```bash
GPT-4_Facial_Expressions/
├── Images/                     # Source images grouped by condition
│   ├── Matrix_inverted/
│   ├── Matrix_upright/
│   ├── Single_inverted/
│   └── Single_upright/
├── Results/                    # GPT-4 responses and answer keys
│   ├── gpt4_image_responses_*.csv
│   ├── SingleFace_answers.csv
│   ├── MatrixFace_answers.csv
│   └── FinalResults/
│       ├── final_results.csv
│       ├── hypothesis_summary.csv
│       ├── accuracy_by_face_type.png
│       ├── accuracy_by_orientation.png
│       └── GPT4_Analysis_Report.pdf
├── GPT-4_Image_Processing.py             # Runs GPT-4 on image folders
├── ImageRotator.py                       # Rotates images in a folder
├── CSV_AnswerKey_Cleaner.py              # Cleans the CSV file with the emotions
├── WSEFEP - norms & FACS - Arkusz1.csv   # Contains the emotion connected to each image
├── ResultsAnalysis.py                    # Statistical analysis + PDF generation
├── .env                                  # Contains your OpenAI API key
└── README.md                             # This file

## Required Libraries for Reproduction

To successfully reproduce this study, you must:

- Obtain an OpenAI API key and store it in a `.env` file.
- Install the following Python libraries in your environment:

| Library         | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `openai`        | Access to GPT-4 Vision model via OpenAI API                             |
| `python-dotenv` | Load the `.env` file that contains your OpenAI API key                  |
| `pandas`        | Data manipulation and merging GPT results with answer keys              |
| `matplotlib`    | Visualization of accuracy by group/orientation                          |
| `seaborn`       | Statistical visualizations with confidence intervals                    |
| `scipy`         | Statistical testing (t-tests, z-tests)                                  |
| `statsmodels`   | Proportions z-test and statistical modeling support                     |
| `fpdf`          | PDF generation for final analysis report                                |
| `Pillow`        | Automated image rotation                                                |
| `scikit-learn`  | (Optional) For advanced analysis like confusion matrices or classifiers |

To install everything at once:

```bash
pip install openai python-dotenv pandas matplotlib seaborn scipy statsmodels fpdf Pillow scikit-le
