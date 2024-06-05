import os
from PIL import Image, ImageDraw, ImageFont

# Set the directory containing the JPG files
directory = ""

# Create a list of all JPG files in the directory
jpg_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]

# Calculate the number of rows and columns needed for the grid
num_files = len(jpg_files)
cols = 5  # Change this value to adjust the number of columns
rows = (num_files + cols - 1) // cols  # Calculate the required number of rows

# Set the font for the labels
font = ImageFont.truetype("path/to/font.ttf", 16)

# Open the first image to get the size
base_image = Image.open(os.path.join(directory, jpg_files[0]))
width, height = base_image.size

# Create a new blank image for the grid
grid_width = cols * width
grid_height = rows * height
grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

# Paste the images into the grid and add labels
for i, filename in enumerate(jpg_files):
    file_path = os.path.join(directory, filename)
    image = Image.open(file_path)
    col = i % cols
    row = i // cols
    x = col * width
    y = row * height
    grid_image.paste(image, (x, y))

    # Add the filename label
    draw = ImageDraw.Draw(grid_image)
    label_width, label_height = draw.textsize(filename, font)
    label_x = x + (width - label_width) // 2
    label_y = y + height + 5
    draw.text((label_x, label_y), filename, font=font, fill=(0, 0, 0))

# Display the grid image
grid_image.show()
