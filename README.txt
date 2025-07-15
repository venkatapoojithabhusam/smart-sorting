✅ README.md
markdown
Copy
Edit
# 🍎 Smart Sorting - Identifying Fresh and Rotten Fruits & Vegetables

This project uses **Deep Learning (MobileNetV2)** and **Streamlit** to identify whether a fruit or vegetable image is **fresh or rotten**.

---

## 📁 Project Structure

Smart Sorting/
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── model/
│ └── fruit_classifier.h5 # Trained model (generated after training)
├── data/
│ ├── fresh/ # Folder with fresh fruit/vegetable images
│ └── rotten/ # Folder with rotten fruit/vegetable images
├── venv/ # Python virtual environment (optional)
└── README.md # Execution instructions

yaml
Copy
Edit

---

## 🛠️ Setup Instructions

### 1. ✅ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
2. ✅ Install Required Packages
bash
Copy
Edit
pip install streamlit tensorflow pillow numpy
🧠 Model Training
3. ✅ Prepare Dataset
Organize your dataset inside the data/ folder like this:

kotlin
Copy
Edit
data/
├── fresh/
│   ├── fresh1.jpg
│   ├── fresh2.jpg
├── rotten/
│   ├── rotten1.jpg
│   ├── rotten2.jpg
4. ✅ Train the Model
Run the training script:

bash
Copy
Edit
python train_model.py
This will save the trained model as:

bash
Copy
Edit
model/fruit_classifier.h5
🚀 Running the Streamlit App
Once the model is trained:

bash
Copy
Edit
streamlit run app.py