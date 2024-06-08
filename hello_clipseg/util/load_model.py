from typing import Any, Tuple
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.modeling_utils import PreTrainedModel


def get_clipseg_model_and_processor() -> Tuple[
    Tuple[
        Any | CLIPSegForImageSegmentation,
        dict[str, Any] | dict[str, Any | list] | Any,
    ]
    | Any
    | CLIPSegForImageSegmentation,
    Tuple[CLIPSegProcessor, dict[str, Any]] | CLIPSegProcessor,
]:
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model: PreTrainedModel = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )

    return model, processor
