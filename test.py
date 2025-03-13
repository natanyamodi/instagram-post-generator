from gradio_client import Client, handle_file
import os

client = Client("gokaygokay/Florence-2")

# Replace 'local_image.jpg' with the actual path to your local image file.
local_image_path = 'cake.jpeg'

# Check if the file exists
if not os.path.exists(local_image_path):
    print(f"Error: File not found at {local_image_path}")
else:
    try:
        result = client.predict(
            image=handle_file(local_image_path),
            task_prompt="Detailed Caption",
            text_input=None,
            model_id="microsoft/Florence-2-large",
            api_name="/process_image"
        )
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")