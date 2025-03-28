import os
from PIL import Image


def combine_images_horizontally(input_folder, output_path):
    """
    Load all images from a folder, combine them horizontally, and save the result.

    Args:
        input_folder (str): Path to the folder containing images
        output_path (str): Path where the combined image will be saved
    """
    # Get all image files from the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Sort the filenames to ensure consistent ordering
    image_files.sort()

    # Load all images
    images = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        try:
            img = Image.open(img_path)
            images.append(img)
            print(f"Loaded: {img_file}, Size: {img.size}")
        except Exception as e:
            print(f"Error loading {img_file}: {e}")

    if not images:
        print("Could not load any images.")
        return

    # Calculate the total width and maximum height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste each image next to each other
    current_width = 0
    for img in images:
        # Calculate vertical position (center the image if shorter than max_height)
        y_position = (max_height - img.height) // 2

        combined_image.paste(img, (current_width, y_position))
        current_width += img.width

    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved to: {output_path}")
    print(f"Final image size: {combined_image.size}")


# Example usage
if __name__ == "__main__":
    # Replace with your folder path and desired output filename
    input_folder = r"C:\Users\20203226\Documents\GitHub\8DM20_Capita_Selecta\figures"
    output_path = input_folder + "\combined\combined_images.jpg"

    combine_images_horizontally(input_folder, output_path)