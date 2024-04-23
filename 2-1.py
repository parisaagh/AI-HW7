import os
from shutil import copyfile

# Define paths to the original dataset and the split dataset
original_dataset_dir = 'C:\\Users\\Rajabi\\Desktop\\NASA\\RSSCN7-master'
base_dir = 'C:\\Users\\Rajabi\\Desktop\\NASA\\RSSCN7-master\\New'

# Create base directory
os.makedirs(base_dir, exist_ok=True)

# Directories for training, validation, and test sets
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Define classes (folders) in the dataset
classes = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking']

# Define the number of images for each set
num_images_per_class = 400
num_train_images = int(0.7 * num_images_per_class)
num_validation_images = int(0.15 * num_images_per_class)

# Copy images to the corresponding directories (train, validation, test)
for class_name in classes:
    # Create subdirectories in train, validation, and test directories
    train_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)

    validation_class_dir = os.path.join(validation_dir, class_name)
    os.makedirs(validation_class_dir, exist_ok=True)

    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy images to train, validation, and test directories based on the split
    class_images = os.listdir(os.path.join(original_dataset_dir, class_name))
    for i, image_name in enumerate(class_images):
        if i < num_train_images:
            copyfile(os.path.join(original_dataset_dir, class_name, image_name),
                     os.path.join(train_class_dir, image_name))
        elif i < num_train_images + num_validation_images:
            copyfile(os.path.join(original_dataset_dir, class_name, image_name),
                     os.path.join(validation_class_dir, image_name))
        else:
            copyfile(os.path.join(original_dataset_dir, class_name, image_name),
                     os.path.join(test_class_dir, image_name))

# Print paths for verification
print("Train Directory:", train_dir)
print("Validation Directory:", validation_dir)
print("Test Directory:", test_dir)
