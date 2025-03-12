import streamlit as st
from main import main_workflow
from PIL import Image
import io

st.write("""# Instagram Post Generator""")
image = st.file_uploader(label="Upload your image", type=["jpg", "jpeg", "png"])
text_input = st.text_input("Optional: Add some context to the image", "")

if image:
    image_bytes = image.getvalue()
    st.image(image_bytes, caption="Uploaded Image")

    if st.button("Generate Post"):
        with st.spinner("Generating Post..."):
            try:
                img = Image.open(io.BytesIO(image_bytes))
                improved_post = main_workflow(img, text_input)

                st.markdown(
                    f"""
                    <style>
                    .stCode pre code {{
                        white-space: pre-wrap !important;
                        overflow-x: hidden !important;
                        width: 100% !important;
                        box-sizing: border-box !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.code(improved_post, language="text")

            except Exception as e:
                st.error(f"An error occurred: {e}")