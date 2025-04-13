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
    "matrix": "What is the average emotion the group is experiencing (anger, surprise, disgust, joy, neutral, fear, or sadness)? Answer by choosing one of the options without explaining further.  The format should be as follow: answer choice, explanation of why you selected it. For example: sadness, The person appears to...",
    "single": "What emotion is the person in the image experiencing (anger, surprise, disgust, joy, neutral, fear, or sadness)? Answer by choosing one of the options. The format should be as follow: answer choice, explanation of why you selected it. For example: sadness, The person appears to..."
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

# Process each image directory
for image_dir in image_dirs:
    category_name = os.path.basename(image_dir).lower()

    # Choose prompt based on folder name
    if "matrix" in category_name:
        prompt = prompts["matrix"]
    elif "single" in category_name:
        prompt = prompts["single"]
    else:
        print(f"Skipping {image_dir} category unresolved.")
        continue

    # Build CSV file path
    output_csv = os.path.join(output_folder, f"gpt4_image_responses_{category_name}.csv")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "response", "full_response"])

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(image_dir, filename)

                try:
                    with open(filepath, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        max_tokens=1000
                    )

                    content = response.choices[0].message.content.strip()
                    emotion_response = content.split(",", 1)[0].strip().lower()
                    writer.writerow([filename, emotion_response, content])
                    print(f"Processed {filename} → {output_csv}")

                except Exception as e:
                    print(f"Error processing {filename} in {category_name}: {str(e)}")

print("Image processing complete. Running analysis:")
Analysis.main()