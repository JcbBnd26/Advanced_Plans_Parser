"""Export annotations in COCO and Pascal VOC formats.

Only *non-delete* corrected detections are exported, giving downstream
object-detection pipelines (YOLO, Detectron2, etc.) clean training data.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from xml.dom import minidom

from .store import CorrectionStore

# Default label → category_id mapping (1-indexed for COCO)
_DEFAULT_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "notes_column", "supercategory": "element"},
    {"id": 2, "name": "header", "supercategory": "element"},
    {"id": 3, "name": "abbreviations", "supercategory": "element"},
    {"id": 4, "name": "legend", "supercategory": "element"},
    {"id": 5, "name": "revision", "supercategory": "element"},
    {"id": 6, "name": "standard_detail", "supercategory": "element"},
    {"id": 7, "name": "title_block", "supercategory": "element"},
    {"id": 8, "name": "misc_title", "supercategory": "element"},
]
_LABEL_TO_CAT: dict[str, int] = {c["name"]: c["id"] for c in _DEFAULT_CATEGORIES}


def _get_corrected_detections(store: CorrectionStore) -> List[dict]:
    """Return all non-delete corrected detections with their final labels.

    For each detection that has corrections, uses the *most recent*
    non-delete correction as the ground-truth label and bbox.
    """
    rows = store._conn.execute(
        "SELECT c.detection_id, c.doc_id, c.page, "
        "       c.corrected_element_type AS label, "
        "       c.corr_bbox_x0, c.corr_bbox_y0, c.corr_bbox_x1, c.corr_bbox_y1, "
        "       d.text_content "
        "FROM corrections c "
        "JOIN detections d ON c.detection_id = d.detection_id "
        "WHERE c.correction_type != 'delete' "
        "  AND c.correction_id = ("
        "      SELECT c2.correction_id FROM corrections c2 "
        "      WHERE c2.detection_id = c.detection_id "
        "        AND c2.correction_type != 'delete' "
        "      ORDER BY c2.corrected_at DESC LIMIT 1"
        "  )"
    ).fetchall()

    results: list[dict] = []
    for r in rows:
        results.append(
            {
                "detection_id": r["detection_id"],
                "doc_id": r["doc_id"],
                "page": r["page"],
                "label": r["label"],
                "bbox": (
                    r["corr_bbox_x0"],
                    r["corr_bbox_y0"],
                    r["corr_bbox_x1"],
                    r["corr_bbox_y1"],
                ),
                "text_content": r["text_content"] or "",
            }
        )
    return results


def export_coco(
    store: CorrectionStore,
    output_dir: Path,
) -> Path:
    """Export corrected annotations in COCO JSON format.

    Creates ``output_dir/annotations.json`` with ``images``,
    ``annotations``, and ``categories`` arrays.

    Parameters
    ----------
    store : CorrectionStore
        Open database connection.
    output_dir : Path
        Output directory (created if needed).

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.json"

    detections = _get_corrected_detections(store)

    # Build image entries (one per unique doc_id+page)
    image_map: dict[tuple[str, int], int] = {}
    images: list[dict] = []

    # Fetch page dimensions from most recent detection per page
    for det in detections:
        key = (det["doc_id"], det["page"])
        if key not in image_map:
            img_id = len(images) + 1
            image_map[key] = img_id

            # Try to get page dims from document's detections
            dim_row = store._conn.execute(
                "SELECT d.bbox_x1, d.bbox_y1 FROM detections d "
                "WHERE d.doc_id = ? AND d.page = ? "
                "ORDER BY d.bbox_x1 DESC LIMIT 1",
                (det["doc_id"], det["page"]),
            ).fetchone()

            images.append(
                {
                    "id": img_id,
                    "file_name": f"{det['doc_id']}_page{det['page']}.pdf",
                    "width": float(dim_row["bbox_x1"]) if dim_row else 2448.0,
                    "height": float(dim_row["bbox_y1"]) if dim_row else 1584.0,
                }
            )

    # Build annotation entries
    annotations: list[dict] = []
    for i, det in enumerate(detections, 1):
        x0, y0, x1, y1 = det["bbox"]
        w = x1 - x0
        h = y1 - y0
        cat_id = _LABEL_TO_CAT.get(det["label"], 0)
        if cat_id == 0:
            continue  # unknown label

        img_id = image_map.get((det["doc_id"], det["page"]), 0)
        annotations.append(
            {
                "id": i,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            }
        )

    coco = {
        "info": {
            "description": "Plan element annotations",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": _DEFAULT_CATEGORIES,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2, ensure_ascii=False)

    return output_path


def export_voc(
    store: CorrectionStore,
    output_dir: Path,
) -> Path:
    """Export corrected annotations in Pascal VOC XML format.

    Creates one XML file per (doc_id, page) pair in *output_dir*.

    Parameters
    ----------
    store : CorrectionStore
        Open database connection.
    output_dir : Path
        Output directory (created if needed).

    Returns
    -------
    Path
        The output directory containing the XML files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = _get_corrected_detections(store)

    # Group by (doc_id, page)
    grouped: Dict[tuple[str, int], list[dict]] = {}
    for det in detections:
        key = (det["doc_id"], det["page"])
        grouped.setdefault(key, []).append(det)

    for (doc_id, page), dets in grouped.items():
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "plan_pages"

        safe_id = doc_id.replace(":", "_")[:40]
        filename = f"{safe_id}_page{page}.pdf"
        ET.SubElement(root, "filename").text = filename

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "2448"
        ET.SubElement(size, "height").text = "1584"
        ET.SubElement(size, "depth").text = "3"

        for det in dets:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = det["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            x0, y0, x1, y1 = det["bbox"]
            ET.SubElement(bndbox, "xmin").text = str(round(x0))
            ET.SubElement(bndbox, "ymin").text = str(round(y0))
            ET.SubElement(bndbox, "xmax").text = str(round(x1))
            ET.SubElement(bndbox, "ymax").text = str(round(y1))

        # Pretty-print XML
        rough = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough)
        pretty = reparsed.toprettyxml(indent="  ", encoding=None)
        # Remove extra XML declaration
        lines = pretty.split("\n")
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        xml_str = "\n".join(lines)

        xml_path = output_dir / f"{safe_id}_page{page}.xml"
        xml_path.write_text(xml_str, encoding="utf-8")

    return output_dir
