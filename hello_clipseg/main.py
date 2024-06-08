import typer
from hello_clipseg.cli import app
from hello_clipseg.util.load_image import get_image_from_url
from hello_clipseg.util.plot import (
    convert_preds_to_image_size,
    overlay_masks_on_image,
    plot_output_masks,
)
from hello_clipseg.util.segment import segment_image


@app.callback()
def main():
    print("Welcome to ClipSeg")


if __name__ == "__main__":
    image = get_image_from_url(
        "https://unsplash.com/photos/8Nc_oQsc2qQ/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjcxMjAwNzI0&force=true&w=640"
    )
    prompts = ["plate", "table", "fruit"]
    mask = segment_image(image=image, prompts=prompts)
    print(convert_preds_to_image_size(mask, original_image_size=image.size).shape)
    plot_output_masks(image=image, preds=mask, prompts=prompts)
    overlay_masks_on_image(image=image, preds=mask, prompts=prompts)
