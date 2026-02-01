import pdfplumber

# Test with different rotation handling
with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    print("=== Default behavior ===")
    print(f"Rotation: {page.rotation}")
    print(f"Width x Height: {page.width} x {page.height}")

    chars = page.chars
    transport = [c for c in chars if c.get("text") == "N" and c["x0"] > 2460]
    if transport:
        n = transport[0]
        print(f"\nLast 'N' in TRANSPORTATION:")
        print(f"  x0={n['x0']:.2f}, x1={n['x1']:.2f}")
        print(f"  Past page edge by: {n['x1'] - page.width:.2f}")

    # Try with rotation normalization disabled
    print("\n=== With rotation disabled ===")

    # Check the underlying matrix
    print("\nChecking character matrix info...")
    for c in transport:
        print(f"  matrix: {c.get('matrix', 'N/A')}")
        print(f"  All keys: {list(c.keys())}")
        break

    # Let's check if the issue is in how pdfplumber handles the rotation
    # by looking at the raw character positions
    print("\n=== Checking coordinate system ===")

    # The mediabox is [0, 0, 1584, 2448] (portrait)
    # The rotation is 270, making it landscape
    # pdfplumber reports width=2448, height=1584

    # With 270° rotation (counterclockwise):
    # - Original x becomes new y (from bottom)
    # - Original y becomes new x (from left)

    # So original coordinate (x_orig, y_orig) becomes:
    # new_x = y_orig
    # new_y = original_width - x_orig = 1584 - x_orig

    # If text in original was at y_orig = 2474 (past 2448 height),
    # after rotation it would appear at new_x = 2474 (past 2448 width)

    # This means the PDF AUTHOR placed text past the page boundary in the original coords

    print("The PDF creator placed TRANSPORTATION text extending past the original")
    print("page height (2448) at position 2474. After 270° rotation, this becomes")
    print("x=2474 which is past the rotated page width (2448).")
    print()
    print("This is a PDF creation issue - the text was placed off the canvas.")
