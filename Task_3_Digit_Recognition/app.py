import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# Set page config for a professional look
st.set_page_config(page_title="AI Digit Recognizer", layout="centered")

# 1. LOAD THE PRE-TRAINED BRAIN
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_my_model():
    return tf.keras.models.load_model('mnist_pro_model.h5')

model = load_my_model()

st.title("🖋️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the box below. Our deep learning model will identify it!")

# 2. CREATE THE INTERACTIVE CANVAS
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Here")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=14,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.info("💡 **Tips for best accuracy:**\n"
            "* Draw **1** as a straight vertical line add bar at bottom .\n")
# 3. PREDICTION LOGIC
if canvas_result.image_data is not None:
    # Convert drawn image to grayscale
    img = canvas_result.image_data.astype('uint8')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Process image: Find bounding box and re-center (Matches our training logic)
    coords = cv2.findNonZero(img_gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img_gray[y:y+h, x:x+w]
        img_padded = cv2.copyMakeBorder(img_cropped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        img_final = cv2.resize(img_padded, (28, 28), interpolation=cv2.INTER_AREA)
        
        with col2:
            st.subheader("Prediction")
            # Prepare for CNN: (1, 28, 28, 1)
            img_input = img_final.reshape(1, 28, 28, 1) / 255.0
            prediction = model.predict(img_input)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Display Results
            st.metric(label="Predicted Digit", value=str(digit))
            st.metric(label="Confidence", value=f"{confidence*100:.1f}%")
            st.image(img_final, caption="Resized input (28x28)", width=100)
    else:
        with col2:
            st.info("Start drawing to see the prediction!")