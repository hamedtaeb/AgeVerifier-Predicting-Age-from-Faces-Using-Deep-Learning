## 🧠 AgeVerifier: Predicting Age from Faces Using Deep Learning: https://github.com/hamedtaeb/AgeVerifier-Predicting-Age-from-Faces-Using-Deep-Learning

This project was built as part of a computer vision task for the supermarket chain **Good Seed**, which is exploring ways to enforce legal age restrictions on alcohol sales using facial analysis.

The goal: Build a deep learning model that estimates a person's age based on a facial image captured at checkout, helping to prevent underage alcohol purchases in a fast and scalable way.

---

## 📂 Dataset

- Source: Adapted from ChaLearn's *Looking at People* dataset.
- Total samples: **7,591** labeled facial images.
- Format: 
  - Image folder: `/final_files/`
  - Labels CSV: `/labels.csv` containing `file_name` and `real_age`.

---

## 📊 Exploratory Data Analysis (EDA)

We began with EDA to understand the data distribution and challenges:

1. **Data Loading**  
   - Verified 7,591 samples with no missing values.

2. **Age Distribution Analysis**  
   - Majority of samples were aged **20–40**, with fewer children and elderly.
   - Plotted histogram to visualize imbalance.

3. **Visual Inspection**  
   - Displayed 15 random images across age ranges to assess diversity and image quality.

4. **EDA Insights**  
   - Age imbalance could hurt prediction accuracy for underrepresented groups.
   - Visual noise (e.g., lighting, expression, makeup) may challenge precise prediction.

---

## 🧠 Modeling Approach

The task was formulated as a **regression problem**, predicting a continuous `real_age` value. Here's what we implemented:

### 🔧 Preprocessing
- Used `ImageDataGenerator` for efficient image loading and normalization.
- Resized all images to **224x224 pixels**.
- Applied an **80/20 train-validation split**.

### 🤖 Model Architecture
- Based on **ResNet50** (transfer learning):
  - Pre-trained on ImageNet, `include_top=False`
  - Added `GlobalAveragePooling2D`, `Dense(128)`, `Dropout(0.3)`, and a final `Dense(1)` layer.
- Loss function: `Mean Squared Error`
- Metric: `Mean Absolute Error (MAE)`
- Optimizer: Adam with a learning rate of `0.0001`

### 🏋️‍♀️ Training
- Ran for **20 epochs** on GPU platform
- Best validation MAE achieved: **~6.64 years**
- Training MAE converged to **~3.18 years**

---

## ✅ Conclusion

The model shows promising results in estimating age from facial images, with an average error of ~6–7 years on unseen data. While not yet accurate enough to enforce age-restricted sales (e.g., 17 vs. 18), this lays the foundation for a real-time, scalable age estimation system.

### 🔮 Future Improvements
- Fine-tune the pre-trained ResNet layers.
- Apply data augmentation to improve generalization.
- Explore classification-based approaches (e.g., "under 18?" binary prediction).
- Experiment with more recent models like **EfficientNet** or **ConvNeXt**.

---

## 🚀 Project Structure

```bash
📁 /datasets/faces/
  ├── final_files/          # Image data
  └── labels.csv            # Image filenames + real age

📄 Age prediction notebook.ipynb
📄 project.ipynb             # Script for GPU training
📄 README.md                 # Project overview
