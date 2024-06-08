import plotly.express as px
import plotly.subplots as sp
import torch
from typing import List
from PIL import Image
import numpy as np
from torch.nn import functional as F


def plot_output_masks(
    image: Image.Image, preds: torch.Tensor, prompts: List[str]
) -> None:
    fig = sp.make_subplots(
        rows=1,
        cols=len(prompts) + 1,
        subplot_titles=["Image"] + prompts,
        horizontal_spacing=0.05,
    )

    # Display the main image
    fig.add_trace(px.imshow(image).data[0], row=1, col=1)

    # Display the predicted images
    for i in range(len(prompts)):
        fig.add_trace(
            px.imshow(torch.sigmoid(preds[i][0]).numpy()[::-1, :]).data[0],
            row=1,
            col=i + 2,
        )

    # Update layout
    fig.update_layout(
        height=400,
        width=300 * (len(prompts) + 1),
        title_text="Image and Predicted Masks",
        showlegend=False,
    )

    # Update axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Show the plot
    fig.show()


def convert_preds_to_image_size(
    preds: torch.Tensor, original_image_size: tuple
) -> torch.Tensor:
    # Get the dimensions of the preds matrix
    batch_size, num_classes, height, width = preds.shape

    # Resize the preds matrix to the original image dimensions
    preds_resized = F.interpolate(
        preds,
        size=original_image_size,
        mode="nearest",
        # align_corners=False
    )

    return preds_resized


def overlay_masks_on_image(
    image: Image.Image, preds: torch.Tensor, prompts: List[str]
) -> None:
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Get the original image dimensions
    original_image_size = image_np.shape[:2]

    # Convert preds to the original image size
    preds_resized = convert_preds_to_image_size(preds, original_image_size)

    # Create a subplot for each prompt
    fig = sp.make_subplots(rows=1, cols=len(prompts), subplot_titles=prompts)

    # Iterate over each prompt and its corresponding mask
    for i, (prompt, mask) in enumerate(zip(prompts, preds_resized)):
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
        fig.add_trace(px.imshow(overlaid_image).data[0], row=1, col=i + 1)

    # Update the layout
    fig.update_layout(height=400, width=300 * len(prompts), showlegend=False)

    # Update axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Show the plot
    fig.show()
