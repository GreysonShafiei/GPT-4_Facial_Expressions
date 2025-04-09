import openai
import base64
import os
import csv
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define prompt
prompt = "What emotion is the person in the image experiencing (anger, surprise, disgust, enjoyment, fear, or sadness)?"

# Image directory
image_dir = "Image\Single"

# Output CSV file
output_csv = "gpt4_image_responses_SingleFace.csv"

# Open CSV file and write header
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "response"])  # Add more fields as needed

    # Process each image
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

                # Write row to CSV
                writer.writerow([filename, content])

                print(f"Processed {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
