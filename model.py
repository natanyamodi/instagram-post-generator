import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import tempfile
import os
import gc

@st.cache_resource
def get_gradio_client():
    return Client("gokaygokay/Florence-2")

@st.cache_data
def generate_image_description(image: Image.Image):
    client = get_gradio_client()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name

    try:
        result = client.predict(
            image=handle_file(temp_file_path),
            task_prompt="Detailed Caption",
            text_input=None,
            model_id="microsoft/Florence-2-large",
            api_name="/process_image"
        )
        return result
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    finally:
        os.unlink(temp_file_path)
        gc.collect()