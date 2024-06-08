from typing import List
from typer import Typer

from hello_clipseg.util.load_image import get_image_from_url
from hello_clipseg.util.plot import plot_output_masks
from hello_clipseg.util.segment import segment_image_url

from structlog import getLogger


app = Typer()
logger = getLogger()


@app.command(name="url")
def segment_url(url: str, prompts: List[str]) -> None:
    logger.info(url=url, prompts=prompts)

    image = get_image_from_url(url=url)
    logger.info("Image loaded")

    logger.info("Segmentation started...")
    y = segment_image_url(url=url, prompts=prompts)
    logger.success("Segmentation complete")

    plot_output_masks(image=image, preds=y, prompts=prompts)
