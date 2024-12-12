import os
import random

def delete_extra_images(data_dir, max_images=75):
    class_names = sorted(os.listdir(data_dir))  # Get class names like A, B, C
    for class_name in class_names:
        class_folder = os.path.join(data_dir, class_name)
        
        # Skip if it's not a directory (e.g., hidden files or non-folder entries)
        if not os.path.isdir(class_folder):
            continue
        
        # Get list of image files in the class folder
        image_files = [f for f in os.listdir(class_folder) if f.endswith(".jpg") or f.endswith(".png")]
        
        # If there are more than 'max_images' images, delete the extra ones
        if len(image_files) > max_images:
            # Randomly select the images to delete (except the first 75)
            images_to_delete = random.sample(image_files, len(image_files) - max_images)
            
            for image_name in images_to_delete:
                image_path = os.path.join(class_folder, image_name)
                os.remove(image_path)  # Delete the image
                print(f"Deleted {image_name} from {class_name}")

def rename_images(data_dir):
    class_names = sorted(os.listdir(data_dir))  # Get class names like A, B, C
    for class_name in class_names:
        class_folder = os.path.join(data_dir, class_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(class_folder):
            continue
        
        # Get list of image files in the class folder
        image_files = [f for f in os.listdir(class_folder) if f.endswith(".jpg") or f.endswith(".png")]
        
        # Sort image files to make sure they are renamed in order
        image_files.sort()
        
        # Rename each image
        for idx, image_name in enumerate(image_files, start=1):
            # Define new name (e.g., A1, A2, B1, B2, etc.)
            new_name = f"{class_name}{idx}.jpg"  # Change extension if needed
            old_image_path = os.path.join(class_folder, image_name)
            new_image_path = os.path.join(class_folder, new_name)
            
            # Check if the new name already exists, and skip renaming if it does
            if os.path.exists(new_image_path):
                print(f"Skipping rename for {image_name} as {new_name} already exists.")
                continue
            
            # Rename the file
            os.rename(old_image_path, new_image_path)
            print(f"Renamed {image_name} to {new_name}")

# Example usage
data_dir = "./asl_dataset"  # Replace with the path to your dataset

# Delete extra images to keep only 75 per class
delete_extra_images(data_dir)

# Rename remaining images to A1, A2, B1, B2, etc.
rename_images(data_dir)
