from PIL import ImageDraw


def save_image_from_np(image, filename):
    # image.save(sys.stdout, "PNG")

    pass


def draw_tracks_on_image(image, tracks):
    draw = ImageDraw.Draw(image)

    for track in tracks:
        detection = track['bbox']

        draw.rectangle(xy=detection, fill=128)

    del draw
    return image
