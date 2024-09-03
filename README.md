# Swin V2 Transformer: Understanding Key Features and Dataset Refinement

##	Problem Statement:
The Swin Transformer model is designed to classify objects, but understanding its reliability and object recognition process is crucial. How does the model determine that it has identified the correct object? What are the key features it uses for classification? Once these critical features are identified, how can we refine the dataset to enhance the model's performance?

##	Title: 
Enhancing Object Classification in the Swin Transformer: Understanding Key Features and Dataset Refinement.

##	Data Sources and Load Dataset:  
The dataset used for this analysis is the ImageNet-1K dataset, available on Hugging Face. ImageNet-1K dataset comprises over 1.2 million high-resolution images across 1,000 diverse classes, including objects, animals, and scenes, and is a benchmark for image classification tasks. Originally developed for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), it is renowned for its quality, featuring human-verified labels that provide reliable ground truth data. Its diversity makes it ideal for training and evaluating deep learning models like the Swin Transformer. However, the dataset has limitations, including a need for more images to capture a broader range of variations and recent developments. Despite this, its accessibility allows for seamless integration into machine-learning workflows.

##	Import Model and Image Processor:
For this data analysis, swinv2 transformer is used. The model is imported from Hugging Face using the identifier “microsoft/swinv2-tiny-patch4-window8-256”.

##	Data Manipulation:
The data instances include the following fields:
image: A PIL.Image.Image object representing the image.
label: An integer classification label.
While most images are in RGB format, some are in grayscale. To simplify and unify the task, grayscale images have been converted to RGB.

##	Data Preparation: 
The prepare_data function is designed to preprocess images for Swin Transformer models by extracting and preparing data for inference. Here's a brief overview of the process:
•	Image and Label Extraction: The function iterates through the input data, extracting images and true labels, and mapping labels to their human-readable names using model.config.id2label.
•	Image Preprocessing: Each image is processed using the processor function, which handles resizing, normalization, and conversion to PyTorch tensors (return_tensors='pt'), ensuring compatibility with the Swin Transformer model.
•	Device Transfer: The processed tensors are moved to the specified device (CPU or GPU) using .to(device), optimizing them for inference.
•	Collecting Prepared Inputs: Processed inputs are stored in a list for efficient batch processing during inference.
The function returns indices, images, true labels, label names, and processed inputs, ready for inference by the Swin Transformer.

##	Analyzing Data:
###	Model Inference: 
The infer function performs model inference on a dataset: For each input, the model predicts labels by selecting the highest logits value, then stores these predictions and their names.

###	Model Performance Evaluation: 
The model's performance is evaluated using several metrics. The model performs well with an accuracy of 81.1%, precision of 73.4%, recall of 73.7% and F1 Score of 72.3%. The results show that the model has moderate precision and recall, leading to some false positives and false negatives. While it performs reasonably well overall, there is room for improvement in reducing these errors to enhance its accuracy and reliability.

The model correctly classifies 81.1% of objects, meaning about 19% of images are incorrectly classified. With 1,000 classes, interpreting each one can be challenging. To simplify, we will analyze each class individually to better understand the target features. For instance, we will start by examining a specific class, such as the Warplane Class (Class 895).

###	Data Organization: 
The dataset is filtered to include only samples with the label 895. This subset is then structured into a DataFrame using pandas. The DataFrame, warplane_dataframe, includes columns for dataset index, images, true labels, true label names, predicted labels, and predicted label names. This organization allows for detailed analysis of predictions and true labels specifically for the Warplane class.

###	Data Analysis: 
- Correct Predictions/ Unique Values: Rows in the warplane_dataframe where the true label matches the predicted label are filtered to identify correct predictions. Among 50 images, in 38 images the warplane class objects are correctly classified.
- Incorrect Predictions/ Distinct Values: Rows where the true label differs from the predicted label are filtered to analyze incorrect predictions. Among 50 images, in 12 images the warplane class objects are incorrectly classified.

##	Data Visualization: 
•	38 images in which the warplane class objects are correctly classified.
•	12 images in which the warplane class objects have been incorrectly classified.
The visualization alone doesn’t reveal the specific features the model relies on. To gain insights into how the model identifies a warplane and which features it considers important, we will use LIME for explanation.

###  Lime Explanation Visualization: 
LIME (Local Interpretable Model-agnostic Explanations) is a technique that helps interpret machine learning models by approximating their behavior locally around a given prediction. It creates an interpretable model that explains how the original model's prediction is influenced by different features, making it easier to understand what the model focuses on for specific instances.
•	Unique Data Lime Explanation
•	Distinct Data Lime Explanation

Lime explanation visualizations help to understand which parts of the image influence the model's decision, providing insights into the model's behavior for each input image.
-	1st Visualization: This section visualizes the LIME explanation with both positive and negative contributions of features shown. The image boundaries are highlighted based on the LIME mask.
-	2nd Visualization: This visualization shows only the features with positive contributions towards the predicted class, without hiding the rest of the image.
-	3rd Visualization: This variant highlights the positive features while hiding the rest of the image, focusing on the most influential regions.
-	4rth Visualization: This heatmap shows the weight of each segment (superpixel) in influencing the model's prediction. The colors represent the contribution of each segment, following this sequence: dark blue, blue, light blue, white, light orange, orange, dark orange, light red, red, dark red.
  	- Dark blue indicates the most influential parts contributing positively. - White represents a neutral or average influence. - Orange shades indicate moderate negative influences. - Dark red shows the most negative contribution.
     This sequence visually communicates how each part of the image impacts the model’s decision, from highly positive (dark blue) to highly negative (dark red).
The visualization shows that the model reliably identifies a warplane when it detects key features like the front, tail, or wheels of the warplane. However, if any of these features are missing from the image, the model struggles to recognize the object as a warplane.

## Recommendation: 
Since the model's performance is significantly impacted when key features (such as the front, tail, or wheels) are missing, consider augmenting the training data with additional images that include various partial views of warplanes. This could help the model learn to recognize warplanes even when some key features are not present. 
Increase the diversity of the training dataset by including images with varying angles, lighting conditions, and occlusions. This can help the model generalize better and become more resilient to feature variations.

## Conclusion: 
The evaluation of the SwinV2 transformer model on the "warplane" class from the ImageNet 1k dataset shows that the model performs well with key features like the front, tail, or wheels but struggles when these features are missing. This underscores the importance of these features in the model’s recognition. To address this, further testing and refinement are needed. Future work should enhance feature detection, improve robustness to missing features, and diversify the dataset to optimize performance. Additionally, a comprehensive analysis of all classes is required, as the current focus is only on the "warplane" class. The project is ongoing, and further testing and refinement will be crucial for addressing these challenges.

