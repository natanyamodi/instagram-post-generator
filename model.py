from gradio_client import Client, handle_file
from PIL import Image
import tempfile
import os

def generate_image_description(image: Image.Image):
    """
    Generates a caption for an image using the Florence-2 model.

    Args:
        image (PIL.Image.Image): The image to generate a caption for.

    Returns:
        str: The generated caption, or None if an error occurs.
    """
    client = Client("gokaygokay/Florence-2")

    try:
        # Save the PIL Image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file, format="JPEG")
            temp_file_path = temp_file.name

        # Pass the temporary file path to handle_file
        result = client.predict(
            image=handle_file(temp_file_path),
            task_prompt="Detailed Caption",
            text_input=None,
            model_id="microsoft/Florence-2-large",
            api_name="/process_image"
        )

        os.unlink(temp_file_path)

        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None