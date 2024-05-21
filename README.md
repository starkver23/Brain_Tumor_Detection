# Brain Tumor Detection Model 

# Introduction
This project involves the creation and training of a brain tumor detection model using convolutional neural networks (CNNs). The model uses MRI images to classify the presence of brain tumors. The project leverages both custom CNNs and transfer learning with the VGG16 model pre-trained on the ImageNet dataset.

# Prerequisites
Ensure you have the following libraries installed:
  1. TensorFlow
  2. Keras
  3. OpenCV
  4. Scikit-learn
  5. Matplotlib
  6. Tqdm

You can install these libraries using pip if you haven't already:
pip install tensorflow keras opencv-python scikit-learn matplotlib tqdm

# Dataset
The dataset used in this project is the "Brain MRI Images for Brain Tumor Detection" dataset, which can be downloaded from Kaggle.
Link to dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

# Steps to Run the Code
1. Setup Kaggle API
To download the dataset from Kaggle, you need to set up the Kaggle API. Ensure you have your kaggle.json file ready.

from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

2. Download and Extract Dataset
!kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
from zipfile import ZipFile

file_name = "brain-mri-images-for-brain-tumor-detection.zip"
with ZipFile(file_name,'r') as zip:
    zip.extractall()
print('Dataset extracted successfully')

3. Load and Preprocess Images
4. Build the model!

# Steps to Clone the Repository and Set Up the Project
1. Clone the Repository
First, you need to clone the repository to your local machine. You can do this using Git.

git clone https://github.com/your-username/your-repository.git
cd your-repository

Replace https://github.com/your-username/your-repository.git with the URL of your repository.

2. Set Up the Environment
To ensure all necessary packages are installed, you should create a virtual environment. This step is optional but recommended to avoid conflicts with other projects.

Using venv (Python 3.3+)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Using conda
If you prefer using Conda, you can create an environment with:

conda create --name brain_tumor_detection python=3.8
conda activate brain_tumor_detection

3. Install Dependencies
Once the environment is activated, install the necessary packages used in the repository.

pip install tensorflow keras opencv-python scikit-learn matplotlib tqdm

4. Download the Dataset
The dataset used in this project is the "Brain MRI Images for Brain Tumor Detection" from Kaggle. You need to download it using the Kaggle API. Make sure you have your kaggle.json file ready.

5. Run the Code
Now, you can run the Jupyter notebook or the Python scripts provided in the repository.

Running Jupyter Notebook
If your project includes a Jupyter notebook, you can start Jupyter and open the notebook:

jupyter notebook
Navigate to the notebook file (e.g., brain_tumor_detection.ipynb) and run the cells.

Running Python Script
If your project is a Python script, you can run it directly:

python brain_tumor_detection.py

6. Make Predictions (Optional)
If you want to make predictions using the trained model, ensure you have an image to test and use the code provided to load and predict the image.

License
This project is licensed under the MIT License.
Feel free to adjust the documentation based on the specifics of your project and repository.
