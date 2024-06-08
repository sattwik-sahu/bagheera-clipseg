from hello_clipseg import ClipSegImageSegmentor


def main():
    clipseg = ClipSegImageSegmentor(
        name="Xavier",
        prompts=["red cylinder", "blue bucket", "crater on ground", "cubical box"],
        output_path="data/out",
        display_preds=True,
    )

    # clipseg.segment_image_from_local(path="data/img/home2.webp")
    clipseg.segment_image_from_url(
        url="https://foyr.com/learn/wp-content/uploads/2022/05/family-room-in-a-house-1024x683.jpg"
    )


if __name__ == "__main__":
    main()
