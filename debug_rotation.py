import pdfplumber

with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    print(f"Page rotation: {page.rotation}")
    print(f"Page width: {page.width}")
    print(f"Page height: {page.height}")
    print(f"Page bbox: {page.bbox}")
    print(f"Page mediabox: {page.mediabox}")

    # The mediabox is the ORIGINAL page size before rotation
    # For a 270 degree rotation, width and height are swapped
    print()
    print("Analysis:")
    print(f"  If 270° rotation swaps dimensions:")
    print(f"    Original width would be: {page.height}")
    print(f"    Original height would be: {page.width}")

    # Let's check the raw page object
    print()
    print("Checking raw PDF structure...")

    # Access the underlying pdfminer page
    try:
        raw = page.page_obj
        print(f"  Raw page type: {type(raw)}")

        # Get MediaBox from raw
        if hasattr(raw, "mediabox"):
            print(f"  Raw mediabox: {raw.mediabox}")
    except Exception as e:
        print(f"  Error accessing raw: {e}")

    # The issue might be that the text was placed assuming no rotation
    # but pdfplumber is returning rotated coordinates

    print()
    print("Checking if TRANSPORTATION text should be clipped...")

    # In a 270° rotation:
    # - Original PDF x becomes display y (inverted)
    # - Original PDF y becomes display x

    # The text at x=2474 might actually be within bounds in the original coordinate system
    # Let me check what the actual visible page boundary would be

    chars = page.chars
    transport_chars = [
        c
        for c in chars
        if "top" in c and c["top"] > 65 and c["top"] < 85 and c["x0"] > 2360
    ]

    if transport_chars:
        last_n = max(transport_chars, key=lambda x: x["x1"])
        print(f"  Last 'N' position: x1={last_n['x1']:.1f}")
        print(f"  Page width: {page.width}")
        print(f"  Overhang: {last_n['x1'] - page.width:.1f}")

        # Check if the overhang corresponds to a rotation issue
        # In 270° rotation, an error in coordinate transformation could cause this
