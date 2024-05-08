import base64
import json
import logging
import os
import anthropic

# OpenAI API Key
API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# Function to encode the image
def encode_image(image_path: str):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_response(prompt: str, base64_image: str):
    """Get response from OpenAI API."""
    client = anthropic.Anthropic(api_key=API_KEY)

    message = client.messages.create(
        # https://docs.anthropic.com/claude/docs/models-overview
        model="claude-3-opus-20240229",
        # model="claude-3-sonnet-20240229",
        # model="claude-3-haiku-20240307",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )
    return message


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    prompt = (
        """I need your help extracting the code and specifications from the image.
        The code typically consists of more than 4 letters and is composed of uppercase letters, numbers, and dashes, 
        like "SA-12345". The first part of the code must always be composed of uppercase letters 
        (e.g., "SAL" rather than "SA1"), while the second part consists of uppercase letters and numbers. 
        If you encounter a format like "SA 12345" with a space, please convert it to "SA-12345". 
        The specifications usually consist of three sets of pure numbers separated by asterisks, 
        like "123*456*789". I need you to output the corresponding code and specifications. 
        When you cannot find the corresponding content, keep the code empty and the specifications as 0, 
        ensuring that there is always a string output for the code and a three-digit array output for the specifications, 
        formatted as pure JSON with no additional symbols (including ```). Below is an example of the desired output:
{
    "code": "SA12345",
    "spec": [12, 23, 34]
}"""
    )

    image_path = "datasets/sample1.jpg"
    base64_image = encode_image(image_path)
    logging.info(f"Loaded image: {image_path}")

    # get response
    response = get_response(prompt, base64_image)

    print("-" * 30)
    print(response)
    print("-" * 30)

    total_tokens = response.usage.input_tokens + response.usage.output_tokens
    print(f"Total Tokens: {total_tokens}\nInput Tokens: {response.usage.input_tokens}\nOutput Tokens: {response.usage.output_tokens}")
    content = response.content[0].text

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
