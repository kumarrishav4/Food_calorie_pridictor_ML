# Food Identification and Segmentation with Calorie Calculation for Complete Meals

**Course**: IIITD ML CSE343/ECE363
**Author**: \Kumar Rishav
**Year**: 2024
**Status**: Final Submission

## 📘 Abstract

This project presents a machine learning-based system that automates food identification from images and estimates their caloric values. Leveraging deep learning (CNNs) and traditional ML models (MLP, Random Forest, SVM), the system recognizes food types and predicts calorie content using nutritional data.

---

## 📁 Datasets

### 🍱 Food Identification

* **Dataset**: [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101)
* **Details**: 100,000+ labeled images across 101 food categories

### 🔢 Calorie Estimation

* **Dataset**: [Calories in Food Items](https://www.kaggle.com/datasets/kkhandekar/calories-in-food-items-per-100-grams)
* **Details**: Nutritional data (fat, cholesterol, sodium, etc.) for common foods per 100g

---

## 🧪 Methodology

### 🖼️ Image Classification

Models used:

* ✅ Random Forest (15–20% accuracy)
* ❌ SVM (5–10% accuracy)
* ⚠️ Combined model (up to 30% accuracy)
* ⭐ **MLP**: 93.37% accuracy
* 🌟 **CNN**: 98.75% accuracy

Preprocessing:

* Image resized to 224x224
* Normalization
* Data split (70/30)
* Validation split (20% of train set)

### 🧮 Calorie Prediction

Models used:

* Linear Regression
* Support Vector Regressor (SVR)
* K-Nearest Neighbors (KNN)
* ⭐ **Random Forest Regressor** (Best performer)

Metrics evaluated:

* **MSE**, **MAE**, and **R² score**

---

## 🔧 Integration

1. Identify food item using CNN/MLP.
2. Fetch corresponding nutrition values.
3. Predict calorie content using the best-trained regressor (Random Forest).
4. Output: Image classification + calorie estimate.

---

## 📈 Results

### Image Classification

* **MLP**: Training Accuracy ≈ 90%, Validation ≈ 100%
* **CNN**: Validation Accuracy = 98.75%

### Calorie Prediction

* **Random Forest Regressor**:

  * MSE: 2675.82
  * MAE: 29.85
  * R²: 0.903

---

## 🛠️ Technologies Used

* Python (Pandas, NumPy, Scikit-learn, TensorFlow, Keras)
* Matplotlib, Seaborn (for visualization)
* Jupyter Notebook / Python scripts

---

## 🧩 Future Improvements

* Improve dataset diversity for real-world deployment
* Introduce object detection models like YOLOv5
* Add mobile app integration for user-friendly food tracking

---

## 📚 References

1. Kagaya et al., "Food detection and recognition using CNN", 2014.
2. Kaushal et al., "Computer vision and deep learning for nutrition detection", 2024.

---

