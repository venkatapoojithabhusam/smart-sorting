# 🍎 Smart Sorting: Transfer Learning for Rotten Fruits/Vegetables Detection

## 🔍 Objective
Classify images of fruits/vegetables as **Fresh** or **Rotten** using MobileNetV2 and Transfer Learning.

## 📁 Folder Structure
```
smart_sorting/
├── dataset/
│   ├── train/
│   │   ├── fresh/
│   │   └── rotten/
│   └── test/
│       ├── fresh/
│       └── rotten/
├── smart_sorting.py
├── requirements.txt
└── README.md
```

## 🛠️ How to Run

1. Clone/download this folder.
2. Add images to the dataset.
3. Install packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the training:
   ```
   python smart_sorting.py
   ```

## 🧠 Output
- Trained model: `rotten_fruit_classifier.h5`

## 📦 Dataset Suggestions
You can use:
- [Kaggle: Fruit and Vegetable Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
