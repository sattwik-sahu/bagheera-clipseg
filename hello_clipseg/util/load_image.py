from PIL import Image
import requests


def get_image_from_local(path: str) -> Image.Image:
    with open(path, "r") as image_file:
        image = Image.open(fp=image_file)
    return image


def get_image_from_url(url: str) -> Image.Image:
    image = Image.open(requests.get(url, stream=True).raw)
    return image
