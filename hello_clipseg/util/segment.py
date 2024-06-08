from email.mime import image
from typing import List

import torch
import typer
from hello_clipseg.util.load_model import get_clipseg_model_and_processor
from hello_clipseg.util.load_image import get_image_from_url
from PIL import Image


app = typer.Typer()
model, processor = get_clipseg_model_and_processor()


def segment_image(image: Image.Image, prompts: List[str]) -> torch.Tensor:
    X = processor(
        text=prompts,
        images=[image] * len(prompts),
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.__call__(**X)
    y = output.logits.unsqueeze(1)

    return y

3
@app.command("url")
def segment_image_url(url: str, prompts: List[str]) -> torch.Tensor:
    image = get_image_from_url(url=url)
    return segment_image(image=image, prompts=prompts)
