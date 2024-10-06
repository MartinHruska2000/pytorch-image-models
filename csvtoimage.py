import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Create output directories if they don't exist
os.makedirs('MNIST/train', exist_ok=True)
os.makedirs('MNIST/test', exist_ok=True)


def save_images_from_csv(csv_file, output_dir):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Check if the label is in the first column, adjust if necessary
    labels = df.iloc[:, 0]  # Assuming the first column contains the label
    pixel_data = df.iloc[:, 1:]  # Remaining columns are pixel values

    # Iterate over each row (image) in the dataframe
    for index, (label, image_data) in tqdm(enumerate(zip(labels, pixel_data.values)), total=df.shape[0]):
        # Reshape the pixel data into a 28x28 image (MNIST format)
        image_array = np.array(image_data, dtype=np.uint8).reshape(28, 28)

        # Convert the NumPy array into a PIL Image
        image = Image.fromarray(image_array)

        # Create a folder for each label (0-9) if it doesn't exist
        label_folder = os.path.join(output_dir, str(label))
        os.makedirs(label_folder, exist_ok=True)

        # Save the image in the specified format (.png, .jpg, .jpeg)
        image.save(os.path.join(label_folder, f'{index}.png'))  # Change extension to .jpg or .jpeg as needed


# Convert the training and test datasets
save_images_from_csv('MNIST/train/mnist_train.csv', 'MNIST/train')
save_images_from_csv('MNIST/test/mnist_test.csv', 'MNIST/test')

print("Image conversion completed!")