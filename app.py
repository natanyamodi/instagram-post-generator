import streamlit as st
from main import main_workflow
from PIL import Image
import io
import pillow_heif

st.write("# Instagram Post Generator")

image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "heic"])
text_input = st.text_input("Optional: Add some context to the image", "")

def convert_heic_to_jpg(image_bytes):
    """Convert HEIC to JPEG using pillow-heif."""
    heif_file = pillow_heif.read_heif(image_bytes)  # Read bytes, not BytesIO
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data
    )
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes

if image:
    image_bytes = image.getvalue()

    # Convert HEIC if needed
    if image.type == "image/heic":
        image_bytes = convert_heic_to_jpg(image_bytes).getvalue()  # Extract bytes after conversion

    img = Image.open(io.BytesIO(image_bytes))  # Open converted bytes
    st.image(img, caption="Uploaded Image")

    if st.button("Generate Post"):
        with st.spinner("Generating Post..."):
            try:
                improved_post = main_workflow(img, text_input)
                st.markdown(
                    f"""
                    <style>
                    pre {{
                        white-space: pre-wrap !important;
                        word-wrap: break-word !important;
                        overflow-x: hidden !important;
                    }}
                    </style>
                    <pre>{improved_post}</pre>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
