import base64
import json
import logging
import os

import requests

# OpenAI API Key
API_KEY = os.environ.get("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path: str):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_response(prompt: str, base64_image: str):
    """Get response from OpenAI API."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    prompt = (
        """There are numbers and specifications in the picture. Can you help me extract them?
        The numbers look like SA12345, and the specifications look like 12*23*34. 
        I need you to format your response as JSON(don't contains the ```), like this:
{
    "number": "SA12345",
    "spec": [12, 23, 34]
}"""
    )

    image_path = "datasets/sample1.jpg"
    base64_image = encode_image(image_path)
    logging.info(f"Loaded image: {image_path}")

    # get response
    response_from_openai = get_response(prompt, base64_image)

    print("-" * 30)
    print(response_from_openai)
    print("-" * 30)

    total_tokens = response_from_openai["usage"]["total_tokens"]
    print("Total Tokens: ", total_tokens)
    content = response_from_openai["choices"][0]["message"]["content"]

    try:
        res = json.loads(content)
    except json.JSONDecodeError:
        res = {}

    print("-" * 30)
    print(f"Response(plain text): {content}")
    print(f"JSON: {json.dumps(res, indent=4)}")
    print("-" * 30)


if __name__ == "__main__":
    main()
