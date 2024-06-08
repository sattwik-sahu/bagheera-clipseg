from datetime import time
from os import path
from typing import List, Optional

import numpy as np
import torch
from torch.nn import functional as F
from hello_clipseg.util.load_model import get_clipseg_model_and_processor
from hello_clipseg.util.load_image import get_image_from_local, get_image_from_url
from PIL import Image

import numpy as np
import plotly.express as px
import plotly.subplots as sp


class ClipSegImageSegmentor:
    def __init__(
        self,
        name: Optional[str],
        prompts: List[str],
        output_path: Optional[str] = None,
        display_preds: bool = True,
    ) -> None:
        self.prompts = prompts
        self.model, self.processor = get_clipseg_model_and_processor()
        self.name = name
        self.output_path = output_path
        self.display_preds = display_preds

        try:
            assert (self.output_path is not None) or self.display_preds
        except AssertionError as ex:
            print(
                f"[ERROR] Both output path = `None` and display_preds = False not allowed together"
            )

    def __resize_ouptuts(self, preds: torch.Tensor, target_size: tuple) -> torch.Tensor:
        # Get the dimensions of the preds matrix
        batch_size, num_classes, height, width = preds.shape

        # Resize the preds matrix to the original image dimensions
        preds_resized = F.interpolate(
            preds, size=target_size, mode="bilinear", align_corners=False
        )

        return preds_resized

    def __segment_image(self, image: Image.Image) -> torch.Tensor:
        X = self.processor(
            text=self.prompts,
            padding=True,
            images=[image] * len(self.prompts),
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.model.__call__(**X)
        y = output.logits.unsqueeze(1)

        preds = self.__resize_ouptuts(preds=y, target_size=image.size)

        fig = self.__overlay_masks_on_image(image=image, preds=preds)
        if self.display_preds:
            fig.show()
        if self.output_path is not None:
            fpath = path.join(
                self.output_path, time.strftime(f"{self.name}__Y-m-d_HMZ.png")
            )
            with open(fpath, "w") as f:
                fig.write_image(file=f, format="png")

        return y

    def segment_image_from_url(self, url: str) -> torch.Tensor:
        image = get_image_from_url(url=url)
        return self.__segment_image(image=image)

    def segment_image_from_local(self, path: str) -> torch.Tensor:
        image = get_image_from_local(path=path)
        return self.__segment_image(image=image)

    def __overlay_masks_on_image(self, image: Image.Image, preds: torch.Tensor):
        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Get the original image dimensions
        original_image_size = image_np.shape[:2]

        # Convert preds to the original image size
        # preds_resized = self.__convert_preds_to_image_size(preds, original_image_size)
        preds_resized = preds

        # Create a subplot for each prompt
        fig = sp.make_subplots(
            rows=1, cols=len(self.prompts), subplot_titles=self.prompts
        )

        # Iterate over each prompt and its corresponding mask
        for i, (prompt, mask) in enumerate(zip(self.prompts, preds_resized)):
            # Apply sigmoid activation to the mask
            sigmoid_mask = torch.sigmoid(mask).squeeze().numpy()

            # Normalize the mask to the range [0, 1]
            normalized_mask = (sigmoid_mask - sigmoid_mask.min()) / (
                sigmoid_mask.max() - sigmoid_mask.min()
            )

            # Create an RGBA mask
            rgba_mask = np.zeros(
                (original_image_size[0], original_image_size[1], 4), dtype=np.float32
            )
            rgba_mask[..., 0] = 1.0  # Red channel
            rgba_mask[..., 3] = normalized_mask  # Alpha channel

            # Overlay the mask on the image
            overlaid_image = np.array(
                image_np * (1 - rgba_mask[..., 3:])
                + 255 * rgba_mask[..., :3] * rgba_mask[..., 3:],
                dtype=np.uint8,
            )

            # Add the overlaid image to the subplot
            fig.add_trace(px.imshow(overlaid_image).data[0][::-1, 1], row=1, col=i + 1)

        # Update the layout
        fig.update_layout(height=400, width=300 * len(self.prompts), showlegend=False)

        # Update axes
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig
