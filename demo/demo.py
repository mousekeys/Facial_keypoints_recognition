import streamlit as st
from PIL import Image, ImageDraw
from src.face_kp.utils.facial_kp_detection import FacialKeyPointDetection
from io import BytesIO

# Initialize FacialKeyPointDetection
facial_key_point_detection = FacialKeyPointDetection()

# Page configuration
st.set_page_config(page_title="Facial Key Point Detection")


# Image uploader
uploaded_image = st.file_uploader('Choose an image (jpg, jpeg, png)', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
   
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True, width=224)
    _, kp = facial_key_point_detection.predict(image)
        
    draw = ImageDraw.Draw(image)
    radius=1
    color=(255, 0, 0)
    for x, y in zip(kp[0], kp[1]):
        # Draw an ellipse around each key point (small circle)
        draw.ellipse(
            [(int(x.item()) - radius, int(y.item()) - radius),
             (int(x.item()) + radius, int(y.item()) + radius)],
            outline=color, fill=color
        )
        
    st.image(image, caption="Processed Image with Key Points", use_column_width=True, width=224)

   