import os
import pandas as pd

# Constants
RESULTS_FOLDER = "Results"
SINGLE_CSV = "SingleFace_answers.csv"
MATRIX_CSV = "MatrixFace_answers.csv"

# Load the data
single_answers = pd.read_csv("WSEFEP - norms & FACS - Arkusz1.csv")

# Keep only Picture ID and Display columns
single_answers = single_answers[["Picture ID", "Display"]]
single_answers["Picture ID"] = single_answers["Picture ID"].str.strip()
single_answers["Display"] = single_answers["Display"].str.strip()

# Rename columns
single_answers = single_answers.rename(columns={
    "Picture ID": "image_name",
    "Display": "correct_emotion"
})

# Save cleaned data
output_path = os.path.join(RESULTS_FOLDER, SINGLE_CSV)
single_answers.to_csv(output_path, index=False)

print(f"Successfully cleaned the file to: {output_path}")