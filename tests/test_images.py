from PIL import Image

from backend.core.images import composite_at_original_resolution, model_size


def test_mask_composite_preserves_original_pixels() -> None:
    original = Image.new("RGB", (32, 24), (10, 20, 30))
    edited = Image.new("RGB", (16, 16), (200, 100, 50))
    mask = Image.new("L", original.size, 0)
    for x in range(10, 20):
        for y in range(8, 16):
            mask.putpixel((x, y), 255)

    result = composite_at_original_resolution(original, edited, mask, feather=0)

    assert result.size == original.size
    assert result.getpixel((0, 0)) == original.getpixel((0, 0))
    assert result.getpixel((12, 10)) == (200, 100, 50)


def test_model_size_keeps_aspect_and_pixel_budget() -> None:
    width, height = model_size((4000, 3000), 1280)
    assert width % 32 == 0
    assert height % 32 == 0
    assert width * height <= 1024 * 1024
    assert abs((width / height) - (4 / 3)) < 0.05
