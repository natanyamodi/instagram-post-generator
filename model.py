import textwrap
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

device = 'cpu'

model_id = 'microsoft/Florence-2-base'

#load Florence-2 model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map = device, trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(model_id, device_map = device, trust_remote_code=True)
     

def generate_image_description(image, text_input=None):
    """Generates a detailed caption for an image."""
    task_prompt = '<MORE_DETAILED_CAPTION>'

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    output = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    long_caption = list(output.values())[0]
    return '\n'.join(textwrap.wrap(long_caption))
