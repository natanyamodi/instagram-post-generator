from langchain_google_genai import ChatGoogleGenerativeAI
from utils.utils import get_api_key

def create_post_content(refined_description: str, text_input: str = None):
    """Creates Instagram post content (caption and hashtags) using Gemini."""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=get_api_key("google_api_key")
    )

    prompt = f"""
    You are a highly successful content creater.
    You love to write a captivating yet simple Instagram caption and generate 20 trending hashtags that will maximize reach, based on the following image description:
    "{refined_description}"
    """

    if text_input:
        prompt += f"""
    Additionally, the user has provided the following context:
    "{text_input}"
        """

    prompt += """
    Your caption should:
    - Tell a story or evoke an emotion.
    - Be engaging and relatable to an Instagram audience.
    - Highlight the key aspects of the image.
    - Be concise and easy to read.

    Your hashtags should:
    - Be relevant to the image, caption, and target audience.
    - Be a mix of broad and specific hashtags.
    - Be currently trending or popular.

    Output format:
    Caption: [Your captivating caption here]

    Hashtags: #[hashtag1] #[hashtag2] ... #[hashtag15] (hashtags to be in lower case)
    """

    try:
        response = model.invoke(prompt)
        post_content = response.content.strip()
        return post_content
    except Exception as e:
        print(f"Error in create_post_content: {e}")
        return "Error generating post content."