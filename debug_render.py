import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

# Render the PDF page to see what's actually visible
print("Rendering PDF page 2...")

# Convert PDF to image at high DPI
images = convert_from_path(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf",
    first_page=3,  # 1-indexed for pdf2image
    last_page=3,
    dpi=72,  # Standard PDF dpi
)

if images:
    img = images[0]
    print(f"Rendered image size: {img.size}")
    print(f"  Width: {img.size[0]}")
    print(f"  Height: {img.size[1]}")

    # Save a crop of the top-right corner where TRANSPORTATION is
    # Scale coordinates from PDF to image (at 72 dpi, 1:1)
    # TRANSPORTATION is around x=2364 to 2474, y=68 to 80

    # Crop top-right corner
    crop_box = (2300, 50, img.size[0], 100)  # (left, top, right, bottom)
    print(f"\nCropping area: {crop_box}")

    try:
        cropped = img.crop(crop_box)
        cropped.save("debug_transportation_crop.png")
        print(f"Saved crop to debug_transportation_crop.png")
        print(f"Crop size: {cropped.size}")
    except Exception as e:
        print(f"Crop error: {e}")

    # Also save full image for reference
    img.save("debug_page2_full.png")
    print(f"Saved full page to debug_page2_full.png")
