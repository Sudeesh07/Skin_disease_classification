# Skin Disease Detection Using Deep Learning and Computer Vision

## Introduction
Skin disease detection using Deep Learning (DL) and Computer Vision (CV) is a transformative approach that harnesses the power of artificial intelligence and image analysis to enhance the diagnosis and management of skin disorders. By training deep learning models, typically based on convolutional neural networks (CNNs), on extensive datasets of skin images, this technology can classify and identify various skin conditions with remarkable accuracy. This innovation not only accelerates the diagnostic process but also holds great potential in making healthcare more accessible, especially through teledermatology and self-assessment tools, ultimately improving patient outcomes and the field of dermatology.

The convergence of DL and CV in skin disease detection marks a significant step toward more efficient and precise healthcare solutions. With the ability to diagnose common issues like acne and severe conditions like melanoma, this technology offers a path to earlier intervention, better patient care, and increased healthcare accessibility, driving advancements in the intersection of medicine and artificial intelligence.

## Problem Statement
The main objective of this project is to develop an application that can be used to detect various classes of skin diseases using the minimal resources available. The application is to be made in such a way that even the people from rural areas of India are able to access and use the application.

## Related Works
There have been numerous research studies and projects related to skin disease detection using Deep Learning (DL) and Computer Vision (CV). Here are some notable works in this field:

- "Dermatologist-level classification of skin cancer with deep neural networks": A pivotal study published in Nature in 2017, which demonstrated how a deep neural network could achieve dermatologist-level performance in classifying skin cancer by analyzing images. This work has been influential in the development of skin disease detection models.
- "SkinNet: A Deep Learning Framework for Dermatologist-level Skin Lesion Classification": Researchers at Stanford University developed SkinNet, a DL framework specifically designed for dermatologist-level skin lesion classification. Their work highlights the potential of DL in revolutionizing dermatology.
- "Dermoscopic Image Classification with Deep Convolutional Neural Networks": This study showcases the use of CNNs to classify dermatoscopic images of skin lesions. It explores the application of DL techniques to enhance the accuracy of skin disease diagnosis.
- "Deep Learning for Skin Lesion Classification": Several projects and competitions, such as the ISIC (International Skin Imaging Collaboration) Melanoma Classification Challenge, have encouraged the development of DL models for skin lesion classification. These competitions have spurred innovation in the field and led to the development of highly accurate models.
- "Automatic Detection of Skin Diseases in Dermoscopy Images with Ensemble Deep Learning Methods": This research investigates the use of ensemble methods with DL techniques for improved accuracy in skin disease detection. Ensembles combine multiple models to enhance overall performance.
- "Skin Disease Diagnosis with Ensemble Deep Learning Models": This work explores the use of ensemble DL models, combining various neural networks, to enhance the robustness and accuracy of skin disease diagnosis, a significant step towards reliable clinical applications.

These works represent a subset of the extensive research and development efforts in the area of skin disease detection using DL and CV. The field continues to evolve, with ongoing contributions from researchers and the medical community to improve accuracy, scalability, and accessibility in skin disease diagnosis and management.

## Solution/Methodology
Our algorithm for skin disease detection adopts a multi-stage approach:

1. **Data Collection**: A diverse dataset of skin images, encompassing a range of skin conditions, was compiled. We have compiled a dataset that consists of images from the SIIM-ISIC Melanoma Classification Challenge hosted on Kaggle and the Skin Cancer MNIST: HAM10000, and compiled a larger directory containing the subclasses of 9 different skin disease classes.

2. **Preprocessing**: Image preprocessing techniques were employed to enhance the quality and consistency of the dataset. For the purpose of this project, we have used TensorFlow’s Image Augmentor function “datagen” to modify the image’s strain, zoom, horizontal or vertical flip, etc., to provide uniqueness to the images. Since the dataset was very unbalanced in the beginning, we have also implemented an artificial image generator function to create new images of already existing images and make a balanced dataset so that when the model is trained it does not prefer one class over the other. In addition, we have to remove the noise in the dataset, in our case hair, we have to develop a color-based segmentation algorithm to detect and mask the hair pixels and focus only on the disease blob.

3. **Model Development**: We have used multiple models to train the dataset and find the best performing model. The models include a custom CNN Model, ResNet50, MobileNetV2, and EfficientNet. So far from our testing, MobileNet and ResNet have provided the best results with a 70 percent accuracy score.

4. **Evaluation**: The algorithm's performance was evaluated through various metrics, including accuracy, precision, recall, and F1-score, using a hold-out validation set.

5. **Deployment**: The model was deployed as a web-based tool, making it accessible to users for skin disease detection. The model will be inputted as a .h5 file and hosted on the cloud where the predictions will be made to reduce the computation load on the smartphone processor.

### Pseudocode
```python
# Set Up the Environment
# Install required Python packages (Kaggle, TensorFlow, Augmentor)
# Upload Kaggle API token (kaggle.json)
# Download and Unzip Dataset
# Download dataset from Kaggle using Kaggle API
# Unzip downloaded dataset

# Import Necessary Libraries
# Import required Python libraries (TensorFlow, Keras, OpenCV, etc.)

# Load and Preprocess Data
# Set up data augmentation with ImageDataGenerator
# Load training, validation, and test data using flow_from_directory

# Transfer Learning Model (MobileNetV2)
# Create MobileNetV2 model with pre-trained weights (excluding top layer)
# Add global average pooling and a new dense output layer for classification
# Compile the model (optimizer='adam', loss='categorical_crossentropy)

# Fine-Tune Model
# Freeze most MobileNetV2 layers
# Compile the model with a lower learning rate (0.0001) and Adam optimizer
# Train the model on training and validation data for 15 epochs, monitoring accuracy

# Data Analysis
# Analyze class distribution in the training dataset

# Data Augmentation with Augmentor
# Augment training dataset using Augmentor (rotate, contrast, brightness, zoom, flip)
# Save augmented images to a new directory

# Load Augmented Data
# Load augmented data using flow_from_directory

# Create a MobileNetV2 model and freeze its layers
# Compile the model (optimizer='Adam', loss='categorical_crossentropy')
# Train the model on further augmented data for 30 epochs, monitoring accuracy
```

##Workflow


![Skin_cancer_flow](https://github.com/Sudeesh07/Skin_disease_classification/assets/135733667/6c817861-4130-4677-9787-50eecb0aa797)

## Conclusion
In conclusion, the development and implementation of a skin disease detection algorithm utilizing Machine Learning and Computer Vision technologies have opened up new horizons in the realm of healthcare accessibility for rural communities in India. This innovative solution not only bridges the gap in medical resources but also empowers individuals in remote areas to proactively manage their skin health.

Usually, the diagnosis of such diseases costs a lot of money for the people in the rural regions of India. But, by harnessing the potential of artificial intelligence, this technology has significantly improved early disease diagnosis, ultimately leading to more effective treatment and reducing the burden of skin ailments in underserved regions. With its potential for scalability and continuous improvement, this algorithm represents a promising step towards a healthier and more equitable future for all, ensuring that nobody is left behind in the pursuit of well-being.


## References
- [Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056): A pivotal study published in Nature in 2017, which demonstrated how a deep neural network could achieve dermatologist-level performance in classifying skin cancer by analyzing images. This work has been influential in the development of skin disease detection models.

- [SkinNet: A Deep Learning Framework for Dermatologist-level Skin Lesion Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8382232/): Researchers at Stanford University developed SkinNet, a DL framework specifically designed for dermatologist-level skin lesion classification. Their work highlights the potential of DL in revolutionizing dermatology.

- [Dermoscopic Image Classification with Deep Convolutional Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0965997822002629): This study showcases the use of CNNs to classify dermatoscopic images of skin lesions. It explores the application of DL techniques to enhance the accuracy of skin disease diagnosis.

- [Deep Learning for Skin Lesion Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9616944/): Several projects and competitions, such as the ISIC (International Skin Imaging Collaboration) Melanoma Classification Challenge, have encouraged the development of DL models for skin lesion classification. These competitions have spurred innovation in the field and led to the development of highly accurate models.

- [Automatic Detection of Skin Diseases in Dermoscopy Images with Ensemble Deep Learning Methods](https://www.researchgate.net/publication/361969073_Detection_and_classification_of_skin_diseases_with_ensembles_of_deep_learning_networks_in_medical_imaging): This research investigates the use of ensemble methods with DL techniques for improved accuracy in skin disease detection. Ensembles combine multiple models to enhance overall performance.

- [Skin Disease Diagnosis with Ensemble Deep Learning Models](https://www.sciencedirect.com/science/article/pii/S1877050919321295): This work explores the use of ensemble DL models, combining various neural networks, to enhance the robustness and accuracy of skin disease diagnosis, a significant step towards reliable clinical applications.

These references provide valuable insights and information related to the field of skin disease detection using Deep Learning and Computer Vision. You can refer to them for further details and in-depth research on the topic.
