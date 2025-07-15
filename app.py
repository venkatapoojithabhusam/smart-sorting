import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("model/fruit_classifier.h5")


# Title
st.title("🍎 Smart Sorting - Fresh or Rotten?")
st.write("Upload a fruit or vegetable image to classify it as **Fresh** or **Rotten**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match training size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    img_array = preprocess_image(img)
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        st.error("🟠 This looks like a **Rotten** fruit/vegetable.")
    else:
        st.success("🟢 This looks like a **Fresh** fruit/vegetable.")



import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model/fruit_classifier.h5")

# Set title
st.title("🍎 Smart Sorting - Fresh or Rotten?")

# Upload image
uploaded_file = st.file_uploader("Upload a fruit or vegetable image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.error("🔴 Prediction: Rotten")
    else:
        st.success("🟢 Prediction: Fresh")
        '''
        import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model/fruit_classifier.h5")

# Set title
st.title("🍎 Smart Sorting - Fresh or Rotten?")

# Upload image
uploaded_file = st.file_uploader("Upload a fruit or vegetable image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.error("🔴 Prediction: Rotten")
    else:
        st.success("🟢 Prediction: Fresh")
'''