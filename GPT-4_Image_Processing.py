import openai
import base64
import os
import csv
import Results
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define prompts
prompts = {
    "matrix": "What is the average emotion the group is experiencing (anger, surprise, disgust, enjoyment, fear, or sadness)? Answer by choosing one of the options without explaining further.",
    "single": "What emotion is the person in the image experiencing (anger, surprise, disgust, enjoyment, fear, or sadness)? Answer by choosing one of the options without explaining further."
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
        writer.writerow(["image_name", "response"])

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(image_dir, filename)

                try:
                    with open(filepath, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

                    response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]}
                        ],
                        max_tokens=1000
                    )

                    content = response["choices"][0]["message"]["content"]
                    writer.writerow([filename, content])
                    print(f"Processed {filename} → {output_csv}")

                except Exception as e:
                    print(f"Error processing {filename} in {category_name}: {str(e)}")

print("Image processing complete. Running analysis:")
Results.ResultsAnalysis.main()