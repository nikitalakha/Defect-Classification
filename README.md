# Defect Classification

It aims to classify surface defects in stainless steel plates using machine learning techniques. The dataset contained 27 indicators representing the geometric shape of the faults. The main steps included:

- **Data Preparation**: The dataset was split into training and testing sets, ensuring an equal representation of each class in both sets.
- **K-Nearest Neighbor (KNN) Classification**: The KNN algorithm was applied with different K values (1, 3, and 5) to classify the surface defects. Model performance was evaluated using confusion matrices and accuracy scores.
- **Data Normalization**: Min-Max normalization was used to scale the attributes in the training set to the range [0-1], enhancing the model's performance by avoiding feature dominance.
- **Bayes Classifier**: A Bayes classifier with unimodal Gaussian density was built to model the data distribution. Its performance was compared to the KNN classifiers.

**Key Findings**:
- KNN Performance: The classification accuracy and confusion matrices varied for different K values, impacting the sensitivity of the classifications.
- Data Normalization: Min-Max normalization significantly improved the model's accuracy by balancing the feature scales.
- Bayes Classifier: The Bayes classifier demonstrated competitive performance, highlighting the importance of understanding data distribution for effective classification.

**Implications:**
The project provides insights into fault classification and emphasizes the significance of data preprocessing and model evaluation. The chosen model can be deployed for real-world applications in industries for automated defect detection in steel plates, enhancing quality control and safety.

**Limitations:**
- Generalization: The model's generalizability to other datasets and scenarios requires further assessment.
- Data Confidentiality: The confidentiality of the 27 indicators underscores the importance of handling sensitive data in real-world applications.

Overall, this exemplifies the application of machine learning for practical problem-solving and emphasizes the importance of model selection, data normalization, and result analysis in building accurate predictive models.
