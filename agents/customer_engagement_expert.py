from langchain_google_genai import ChatGoogleGenerativeAI
from utils.utils import get_api_key
import re

def review_and_improve_post(post_content: str):
    """Reviews and improves the Instagram post content (caption and hashtags)."""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=get_api_key("google_api_key")
    )

    prompt = f"""
    You are a Customer Engagement Expert and editor, tasked with reviewing and improving Instagram post content (caption and hashtags).
    Your goal is to enhance the post's appeal, clarity, and engagement potential.

    Here is the Instagram post content:

    "{post_content}"

    Please provide a revised version of the post content, focusing on:
    - Making the caption more engaging.
    - Ensuring the hashtags are relevant, trending, and likely to reach the target audience.
    - Improving the overall clarity and readability of the post.
    - Correcting any grammatical errors or awkward phrasing.
    - Optimizing the content for maximum engagement.
    - All hashtags must be lowercase.

    Return ONLY the **final instagram post** without any extra messages or formatting.
    """

    try:
        response = model.invoke(prompt)
        improved_post = response.content.strip()

        return improved_post
    except Exception as e:
        print(f"Error in review_and_improve_post: {e}")
        return "Error reviewing and improving post content."