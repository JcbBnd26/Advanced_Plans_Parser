import pdfplumber
from PIL import Image

# Use pdfplumber's built-in rendering
print("Rendering PDF page 2...")

with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    print(f"Page dimensions: {page.width} x {page.height}")
    print(f"Page rotation: {page.rotation}")

    # Render at 72 DPI (1:1 with PDF coordinates)
    img = page.to_image(resolution=72)

    print(f"Rendered image size: {img.original.size}")

    # Save full image
    img.save("debug_page2_full.png")
    print("Saved debug_page2_full.png")

    # The TRANSPORTATION text should be at x=2364 to 2474, y=68 to 80
    # If the image is 2448 wide and text extends to 2474, it would be cut off
    # OR if pdfplumber clips content, we won't see it

    # Check right edge of image
    pil_img = img.original
    print(f"\nImage actual size: {pil_img.size}")

    # If text truly extends past page, it would be clipped in rendering
    # Let's check what's at the far right edge

    # Crop rightmost 200 pixels of top area
    right_edge = pil_img.crop((pil_img.size[0] - 200, 40, pil_img.size[0], 100))
    right_edge.save("debug_right_edge.png")
    print("Saved debug_right_edge.png - shows rightmost 200px of top area")
