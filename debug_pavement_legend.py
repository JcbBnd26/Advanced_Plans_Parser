"""Debug script to find and visualize the Pavement Legend."""

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

# Open the PDF and search for PAVEMENT LEGEND
pdf_path = "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"

# Focus on page 5 (index 5) which has the clear PAVEMENT LEGEND section
PAGE_NUM = 5

with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[PAGE_NUM]

    print(f"Page {PAGE_NUM} dimensions: {page.width} x {page.height}")

    # Based on analysis:
    # PAVEMENT LEGEND: header at y=885, x=2042 to 2195
    # HEAVY-DUTY CONCRETE PAVEMENT: definition at y=919, x=2172 to 2400 | pattern rect (2043.5, 910.4) to (2115.5, 934.4)
    # LIGHT-DUTY CONCRETE PAVEMENT: definition at y=950, x=2172 to 2394 | pattern rect (2043.5, 941.6) to (2115.5, 965.5)
    # CONCRETE SIDEWALK: definition at y=981, x=2172 | pattern rect (2043.5, 972.7) to (2115.5, 996.7)

    # Legend entries with their pattern bboxes and definition text bboxes
    legend_entries = [
        {
            "name": "HEAVY-DUTY CONCRETE PAVEMENT",
            "pattern_bbox": (2043.5, 910.4, 2115.5, 934.4),
            "definition_bbox": (2172.2, 915.0, 2400.0, 931.0),
        },
        {
            "name": "LIGHT-DUTY CONCRETE PAVEMENT",
            "pattern_bbox": (2043.5, 941.6, 2115.5, 965.5),
            "definition_bbox": (2172.2, 946.0, 2394.4, 962.0),
        },
        {
            "name": "CONCRETE SIDEWALK",
            "pattern_bbox": (2043.5, 972.7, 2115.5, 996.7),
            "definition_bbox": (2172.6, 977.0, 2312.0, 993.0),
        },
        {
            "name": "STRUCTURAL SLAB",
            "pattern_bbox": (2043.5, 1003.9, 2115.5, 1027.8),
            "definition_bbox": (2172.6, 1009.0, 2292.7, 1025.0),
        },
        {
            "name": "BUILDING",
            "pattern_bbox": (2043.5, 1035.0, 2115.5, 1059.1),
            "definition_bbox": (2172.2, 1040.0, 2231.6, 1056.0),
        },
        {
            "name": "DETECTABLE WARNING PANELS",
            "pattern_bbox": (2043.5, 1066.1, 2115.5, 1090.3),
            "definition_bbox": (2172.2, 1071.0, 2372.0, 1087.0),
        },
        {
            "name": "AGGREGATE SURFACING",
            "pattern_bbox": (2043.5, 1097.5, 2115.5, 1121.4),
            "definition_bbox": (2173.1, 1103.0, 2330.3, 1119.0),
        },
        {
            "name": "SODDING",
            "pattern_bbox": (2043.5, 1128.6, 2115.5, 1152.5),
            "definition_bbox": (2172.6, 1134.0, 2231.3, 1150.0),
        },
    ]

    # Render the page at higher resolution for clarity
    SCALE = 2  # 2x resolution (144 DPI)
    img = page.to_image(resolution=72 * SCALE)
    pil_img = img.original.convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    print(f"\nImage size: {pil_img.size}")

    # Region of interest for the crop
    ROI = (2000, 870, 2420, 1010)

    # Draw colored boxes around all elements in the Pavement Legend

    # Draw the PAVEMENT LEGEND header box (purple)
    header_bbox = (2042.5, 885, 2195.8, 900)
    hx0, hy0, hx1, hy1 = [v * SCALE for v in header_bbox]
    draw.rectangle(
        [hx0, hy0, hx1, hy1], outline=(138, 43, 226, 255), width=3
    )  # Purple box for header
    print("Drew purple box around 'PAVEMENT LEGEND:' header")

    # For each legend entry, draw:
    # 1. Red-orange box around the pattern
    # 2. Yellow box around the definition text
    # 3. Green line FROM the definition TO the pattern

    for entry in legend_entries:
        px0, py0, px1, py1 = entry["pattern_bbox"]
        dx0, dy0, dx1, dy1 = entry["definition_bbox"]

        # Scale coordinates
        px0_s, py0_s, px1_s, py1_s = px0 * SCALE, py0 * SCALE, px1 * SCALE, py1 * SCALE
        dx0_s, dy0_s, dx1_s, dy1_s = dx0 * SCALE, dy0 * SCALE, dx1 * SCALE, dy1 * SCALE

        padding = 3 * SCALE

        # Draw RED-ORANGE box around pattern
        draw.rectangle(
            [px0_s - padding, py0_s - padding, px1_s + padding, py1_s + padding],
            outline=(255, 69, 0, 255),  # Red-orange
            width=4,
        )

        # Draw YELLOW box around definition text
        draw.rectangle(
            [dx0_s - padding, dy0_s - padding, dx1_s + padding, dy1_s + padding],
            outline=(255, 215, 0, 255),  # Gold/Yellow
            width=3,
        )

        # Draw GREEN line FROM definition TO pattern
        def_y_center = (dy0_s + dy1_s) / 2
        pattern_y_center = (py0_s + py1_s) / 2

        # Start point: left edge of definition text
        start_x = dx0_s - padding
        start_y = def_y_center

        # End point: right edge of pattern box
        end_x = px1_s + padding
        end_y = pattern_y_center

        # Draw the line
        draw.line(
            [start_x, start_y, end_x, end_y], fill=(0, 200, 0, 255), width=3  # Green
        )

        # Add arrowhead pointing LEFT (toward the pattern)
        arrow_size = 10 * SCALE
        draw.polygon(
            [
                (end_x, end_y),  # Tip of arrow
                (end_x + arrow_size, end_y - arrow_size / 2),  # Upper wing
                (end_x + arrow_size, end_y + arrow_size / 2),  # Lower wing
            ],
            fill=(0, 200, 0, 255),
        )

        print(
            f"Drew boxes for '{entry['name']}': red-orange (pattern), yellow (definition), green line"
        )

    # Draw cyan outline around entire PAVEMENT LEGEND region
    legend_region = (2038, 880, 2420, 1160)
    lx0, ly0, lx1, ly1 = [v * SCALE for v in legend_region]
    draw.rectangle(
        [lx0, ly0, lx1, ly1], outline=(0, 191, 255, 200), width=2
    )  # Cyan outline
    print("Drew cyan outline around entire Pavement Legend region")

    # Save full image
    pil_img.save("debug_pavement_legend_overlay.png")
    print(f"\nSaved full overlay to debug_pavement_legend_overlay.png")

    # Crop to show just the legend region
    crop_box = tuple(
        int(v * SCALE) for v in (ROI[0] - 30, ROI[1] - 10, ROI[2] + 10, ROI[3] + 20)
    )
    cropped = pil_img.crop(crop_box)
    cropped.save("debug_pavement_legend_crop.png")
    print(
        f"Saved cropped region to debug_pavement_legend_crop.png (size: {cropped.size})"
    )
