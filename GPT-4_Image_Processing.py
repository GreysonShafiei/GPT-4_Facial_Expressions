from openai import OpenAI
import base64
import os
import csv
import Results
from dotenv import load_dotenv
import Results.ResultsAnalysis as Analysis

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define prompts
prompts = {
    # Exp stands for explanation
    "matrix_exp":  "What is the average emotion the group is experiencing (anger, surprise, disgust, joy, neutral, fear, or sadness)? "
                   "Answer by choosing one of the options, explanation of why you selected it. "
                   "Format: answer choice, explanation.  For example: sadness, The people appear to …",

    "matrix_noExp":"What is the average emotion the group is experiencing (anger, surprise, disgust, joy, neutral, fear, or sadness)? "
                   "Answer by choosing one of the options only.  Format: answer choice.  For example: sadness",

    "single_exp":  "What emotion is the person in the image experiencing (anger, surprise, disgust, joy, neutral, fear, or sadness)? "
                   "Answer by choosing one of the options, explanation of why you selected it. "
                   "Format: answer choice, explanation.  For example: sadness, The person appears to …",

    "single_noExp":"What emotion is the person in the image experiencing (anger, surprise, disgust, joy, neutral, fear, or sadness)? "
                   "Answer by choosing one of the options only.  Format: answer choice.  For example: sadness"
}

# Define image directories to process
image_dirs = [
    "Images/Matrix_upright",
    "Images/Matrix_inverted",
    "Images/Single_upright",
    "Images/Single_inverted"
]

# Output folder for results
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# Pick which prompt keys belong to a directory
def prompt_keys_for(dir_name: str) -> list[str]:
    """Return the two prompt‑dict keys to use for this directory."""
    name = dir_name.lower()
    if "matrix" in name:
        return ["matrix_exp", "matrix_noExp"]
    if "single" in name:
        return ["single_exp", "single_noExp"]
    return []

for image_dir in image_dirs:
    keys = prompt_keys_for(image_dir)
    if not keys:
        print(f"Skipping {image_dir}: category unresolved.")
        continue

    # Run for the number of prompts in category
    for pkey in keys:
        prompt_text = prompts[pkey]
        category_name = os.path.basename(image_dir).lower()
        output_csv = os.path.join(output_folder, f"gpt4_image_responses_{category_name}_{pkey}.csv")

        with open(output_csv, "w", newline="", encoding="utf‑8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_name", "response", "full_response"])

            for fname in os.listdir(image_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                fpath = os.path.join(image_dir, fname)

                try:
                    with open(fpath, "rb") as img:
                        b64 = base64.b64encode(img.read()).decode()

                    resp = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
                            ]
                        }],
                        max_tokens=500
                    )

                    full = resp.choices[0].message.content.strip()
                    # works for both “label, explanation” and “label”
                    short = full.split(",", 1)[0].strip().lower()
                    writer.writerow([fname, short, full])
                    print(f"[{pkey}] {fname}  →  {output_csv}")

                except Exception as e:
                    print(f"Error processing {fname} ({pkey}): {e}")

print("Image processing complete. Running analysis:")
Analysis.main()