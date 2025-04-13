import os
from PIL import Image

# Input and output folders
input_folder = "Images/Matrix_upright"
output_folder = "Images/Matrix_inverted"

# Loop through all JPGs
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open, rotate, and save
        with Image.open(input_path) as img:
            rotated = img.rotate(180)
            rotated.save(output_path)

print("Successfully rotated the images 180 to: ", output_folder)