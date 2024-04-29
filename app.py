from openai import OpenAI
import streamlit as st
import requests
from io import BytesIO
import base64
import os
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
STABILITY_AI_API_KEY = os.getenv("STABILITY_AI_API_KEY")

client = OpenAI(
    api_key=OPEN_AI_API_KEY
)


def generate_image(prompt):
    """
    Generates an image using the DALL-E model based on the given prompt.

    Args:
        prompt (str): The prompt to generate the image from.

    Returns:
        tuple: A tuple containing the image bytes and the revised prompt.
    """

    AI_Response = client.images.generate(
        model="dall-e-3",
        size="1024x1024",
        quality="hd",
        n=1,
        response_format="url",
        prompt=prompt
    )

    image_url = AI_Response.data[0].url
    revised_prompt = AI_Response.data[0].revised_prompt

    response = requests.get(image_url)
    image_bytes = BytesIO(response.content)

    return image_bytes, revised_prompt



def create_image_variation(source_image_url):
    """
    Creates a variation of the source image with a size of 1024x1024 pixels.

    Args:
        source_image_url (str): The URL or file path of the source image.

    Returns:
        BytesIO: The image variation as BytesIO object.

    Raises:
        FileNotFoundError: If the source image file is not found.
        requests.exceptions.RequestException: If there is an error while retrieving the generated image.

    """
    AI_Response = client.images.create_variation(
        image=open(source_image_url, "rb"),
        size="1024x1024",
        n=1,
        response_format="url"
    )

    generated_image_url = AI_Response.data[0].url

    response = requests.get(generated_image_url)
    image_bytes = BytesIO(response.content)

    return image_bytes


def generate_with_SD(prompt):
    """
    Generates an image using the Stable Diffusion XL 1024 model from the Stability AI API.

    Args:
        prompt (str): The text prompt to generate the image.

    Returns:
        dict: The generated image data in JSON format.
    """

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_AI_API_KEY}",
    }

    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts": [
            {
                "text": prompt,
                "weight": 1
            },
            {
                "text": "blurry, bad",
                "weight": -1
            }
        ],
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )

    data = response.json()

    return data


tab_generate,tab_variation, tab_SD = st.tabs(["Create Image", "Create Variation ", "Stable Diffusion"])

with tab_generate:
    st.subheader("Creating Images with DALL-E 3")
    st.divider()
    prompt = st.text_input("Describe the image you want to generate", key="text_input")
    generate_btn = st.button("Generate Image")

    if generate_btn:
        with st.spinner("Generating image..."):
            image_data, revised_prompt = generate_image(prompt)

            st.image(image=image_data)
            st.divider()
            st.caption(revised_prompt)

with tab_variation:
    st.subheader("Creating Image Variations with DALL-E 3")
    st.divider()
    selected_file = st.file_uploader("Select an image in PNG format", type=["png"])

    if selected_file:
        st.image(image=selected_file.name)

    variation_btn = st.button("Create Variation")

    if variation_btn:
        with st.spinner("Creating image variation..."):
            image_data = create_image_variation(selected_file.name)

            st.image(image=image_data)

with tab_SD:
    st.subheader("Creating Images with Stable Diffusion")
    st.divider()
    SD_prompt = st.text_input("Describe the image you want to generate", key="sd_text_input")
    SD_generate_btn = st.button("Create", key="sd_button")

    if SD_generate_btn:
        with st.spinner("Generating image..."):
            data = generate_with_SD(SD_prompt)


            for image in data["artifacts"]:
                image_bytes = base64.b64decode(image["base64"])
                st.image(image=image_bytes)