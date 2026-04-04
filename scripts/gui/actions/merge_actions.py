"""Merge and link-column actions for the annotation tab.

Functions accept the ``tab`` (AnnotationTab) instance and operate on
its state.  Extracted from EventHandlerMixin.
"""

from __future__ import annotations

from tkinter import messagebox, simpledialog
from typing import Any

from plancheck.analysis.box_merge import merge_boxes, polygon_bbox
from plancheck.corrections.features import featurize_region
from plancheck.ingest.ingest import extract_text_in_bbox, extract_text_in_polygon

from ..annotation_state import CanvasBox


def merge_boxes_action(tab: Any) -> None:
    """Merge multi-selected overlapping boxes, or reshape a detection to
    the enclosing bbox of selected word-overlay rectangles."""

    if len(tab._selected_word_rids) >= 2:
        merge_words_into_detection(tab)
        return

    targets = list(tab._multi_selected)
    if tab._selected_box and tab._selected_box not in targets:
        targets.append(tab._selected_box)

    if len(targets) < 2:
        tab._status.configure(text="Select at least 2 boxes to merge (Shift+Click)")
        return
    if not tab._doc_id:
        tab._status.configure(text="No document loaded")
        return

    types_seen = {cb.element_type for cb in targets}
    if len(types_seen) == 1:
        merged_type = types_seen.pop()
    else:
        largest = max(
            targets,
            key=lambda cb: (
                (cb.pdf_bbox[2] - cb.pdf_bbox[0]) * (cb.pdf_bbox[3] - cb.pdf_bbox[1])
            ),
        )
        merged_type = largest.element_type
        answer = simpledialog.askstring(
            "Merge Type",
            f"Boxes have mixed types: {', '.join(sorted(types_seen))}.\n"
            f"Enter the merged element type (default: {merged_type}):",
            initialvalue=merged_type,
            parent=tab.root,
        )
        if answer is None:
            return
        merged_type = answer.strip() or merged_type

    bboxes = [cb.pdf_bbox for cb in targets]
    merged_poly = merge_boxes(bboxes)
    merged_bbox = polygon_bbox(merged_poly)

    if tab._pdf_path:
        merged_text = extract_text_in_polygon(tab._pdf_path, tab._page, merged_poly)
    else:
        merged_text = "\n".join(cb.text_content for cb in targets if cb.text_content)

    largest = max(
        targets,
        key=lambda cb: (
            (cb.pdf_bbox[2] - cb.pdf_bbox[0]) * (cb.pdf_bbox[3] - cb.pdf_bbox[1])
        ),
    )
    merged_features = largest.features

    survivor = targets[0]
    consumed = targets[1:]

    tab._push_undo(
        "merge",
        survivor,
        extra={
            "merged_boxes": [
                {
                    "detection_id": cb.detection_id,
                    "element_type": cb.element_type,
                    "pdf_bbox": cb.pdf_bbox,
                    "polygon": cb.polygon,
                    "confidence": cb.confidence,
                    "text_content": cb.text_content,
                    "corrected": cb.corrected,
                }
                for cb in targets
            ],
        },
    )

    for cb in consumed:
        tab._store.save_correction(
            doc_id=tab._doc_id,
            page=tab._page,
            correction_type="delete",
            corrected_label=cb.element_type,
            corrected_bbox=cb.pdf_bbox,
            detection_id=cb.detection_id,
            original_label=cb.element_type,
            original_bbox=cb.pdf_bbox,
            session_id=tab._session_id,
        )
        if cb.rect_id:
            tab._canvas.delete(cb.rect_id)
        if cb.label_id:
            tab._canvas.delete(cb.label_id)
        for hid in cb.handle_ids:
            tab._canvas.delete(hid)
        tab._canvas_boxes.remove(cb)
        tab._session_count += 1

    tab._store.save_correction(
        doc_id=tab._doc_id,
        page=tab._page,
        correction_type="reshape",
        corrected_label=merged_type,
        corrected_bbox=merged_bbox,
        detection_id=survivor.detection_id,
        original_label=survivor.element_type,
        original_bbox=survivor.pdf_bbox,
        session_id=tab._session_id,
    )

    survivor.element_type = merged_type
    survivor.pdf_bbox = merged_bbox
    survivor.polygon = merged_poly
    survivor.text_content = merged_text
    survivor.features = merged_features
    survivor.corrected = True
    survivor.merged_from = [cb.detection_id for cb in targets]
    tab._session_count += 1

    tab._store.update_detection_polygon(survivor.detection_id, merged_poly, merged_bbox)

    tab._selected_box = None
    tab._clear_multi_select()
    tab._draw_box(survivor)

    tab._update_session_label()
    tab._update_page_summary()
    tab._status.configure(
        text=f"Merged {len(targets)} boxes \u2192 {merged_type} "
        f"(polygon with {len(merged_poly)} vertices)"
    )


def merge_words_into_detection(
    tab: Any,
    *,
    forced_type: str | None = None,
    force_create: bool = False,
) -> None:
    """Reshape the selected detection to enclose selected word-overlay
    rectangles, or create a new detection from words."""
    if not tab._doc_id:
        tab._status.configure(text="No document loaded")
        return

    word_bboxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    for rid in tab._selected_word_rids:
        winfo = tab._word_overlay_items.get(rid)
        if not winfo:
            continue
        word_bboxes.append((winfo["x0"], winfo["top"], winfo["x1"], winfo["bottom"]))
        if winfo.get("text"):
            texts.append(winfo["text"])

    if not word_bboxes:
        tab._status.configure(text="No valid words selected")
        return

    merged_poly: list[tuple[float, float]] | None = None
    if len(word_bboxes) >= 2:
        try:
            merged_poly = merge_boxes(word_bboxes)
        except Exception:  # noqa: BLE001
            merged_poly = None
    new_bbox = (
        polygon_bbox(merged_poly)
        if merged_poly
        else (
            min(b[0] for b in word_bboxes),
            min(b[1] for b in word_bboxes),
            max(b[2] for b in word_bboxes),
            max(b[3] for b in word_bboxes),
        )
    )

    if tab._pdf_path and merged_poly:
        merged_text = extract_text_in_polygon(tab._pdf_path, tab._page, merged_poly)
    elif tab._pdf_path:
        merged_text = extract_text_in_bbox(tab._pdf_path, tab._page, new_bbox)
    else:
        merged_text = " ".join(texts)

    n_words = len(word_bboxes)

    if tab._selected_box and not force_create and forced_type is None:
        cbox = tab._selected_box
        orig_bbox = cbox.pdf_bbox
        orig_polygon = list(cbox.polygon) if cbox.polygon else None

        tab._push_undo(
            "reshape",
            cbox,
            extra={"orig_bbox": orig_bbox, "orig_polygon": orig_polygon},
        )
        tab._store.save_correction(
            doc_id=tab._doc_id,
            page=tab._page,
            correction_type="reshape",
            corrected_label=cbox.element_type,
            corrected_bbox=new_bbox,
            detection_id=cbox.detection_id,
            original_label=cbox.element_type,
            original_bbox=orig_bbox,
            corrected_text=merged_text,
            session_id=tab._session_id,
        )

        cbox.pdf_bbox = new_bbox
        cbox.polygon = merged_poly
        cbox.text_content = merged_text
        cbox.corrected = True
        tab._session_count += 1

        tab._store.update_detection_polygon(cbox.detection_id, merged_poly, new_bbox)
        tab._draw_box(cbox)
        tab._clear_word_selection()
        tab._update_session_label()
        tab._update_page_summary()
        poly_info = f" (polygon {len(merged_poly)} pts)" if merged_poly else ""
        tab._status.configure(
            text=f"Reshaped {cbox.element_type} to enclose {n_words} words{poly_info}"
        )
    else:
        features = featurize_region("misc_title", new_bbox, None, 2448.0, 1584.0)
        chosen_type = forced_type or (tab._type_var.get() or "misc_title")
        if forced_type is None and features:
            try:
                prediction = tab._predict_model_suggestion(features, text=merged_text)
                if prediction is None:
                    raise ValueError("No configured model is available")
                pred_label, pred_conf, _ = prediction
                if pred_conf and pred_conf > 0.5:
                    chosen_type = pred_label
            except Exception:  # noqa: BLE001
                pass

        det_id = tab._store.save_detection(
            doc_id=tab._doc_id,
            page=tab._page,
            run_id=tab._run_id or "manual",
            element_type=chosen_type,
            bbox=new_bbox,
            text_content=merged_text,
            features=features,
        )
        tab._store.save_correction(
            doc_id=tab._doc_id,
            page=tab._page,
            correction_type="add",
            corrected_label=chosen_type,
            corrected_bbox=new_bbox,
            detection_id=det_id,
            session_id=tab._session_id,
        )

        cbox = CanvasBox(
            detection_id=det_id,
            element_type=chosen_type,
            confidence=None,
            text_content=merged_text,
            features=features,
            pdf_bbox=new_bbox,
            polygon=merged_poly,
            corrected=True,
        )
        if merged_poly:
            tab._store.update_detection_polygon(det_id, merged_poly, new_bbox)
        tab._canvas_boxes.append(cbox)
        tab._draw_box(cbox)
        tab._select_box(cbox)
        tab._session_count += 1
        tab._update_session_label()
        tab._update_page_summary()
        tab._clear_word_selection()
        poly_info = f" (polygon {len(merged_poly)} pts)" if merged_poly else ""
        tab._status.configure(
            text=f"Created {chosen_type} from {n_words} words{poly_info}"
        )


def link_column_action(tab: Any) -> None:
    """Create a notes_column detection that encloses selected
    header / notes_block boxes, then group them under it."""
    if not tab._doc_id:
        return

    targets = list(tab._multi_selected)
    if tab._selected_box and tab._selected_box not in targets:
        targets.append(tab._selected_box)

    if len(targets) < 2:
        tab._status.configure(
            text="Shift+click \u22652 headers / notes blocks, then press L"
        )
        return

    linkable_types = {"header", "notes_block"}
    non_linkable = [cb for cb in targets if cb.element_type not in linkable_types]
    if non_linkable:
        bad = ", ".join(sorted({cb.element_type for cb in non_linkable}))
        messagebox.showwarning(
            "Invalid Selection",
            f"Only header and notes_block boxes can be linked "
            f"into a notes column.\n\nFound: {bad}",
        )
        return

    already_grouped = [cb for cb in targets if cb.group_id]
    if already_grouped:
        tab._status.configure(text=f"{len(already_grouped)} box(es) already in a group")
        return

    x0 = min(cb.pdf_bbox[0] for cb in targets)
    y0 = min(cb.pdf_bbox[1] for cb in targets)
    x1 = max(cb.pdf_bbox[2] for cb in targets)
    y1 = max(cb.pdf_bbox[3] for cb in targets)
    col_bbox = (x0, y0, x1, y1)

    text_content = ""
    if tab._pdf_path:
        text_content = extract_text_in_bbox(tab._pdf_path, tab._page, col_bbox)

    features = featurize_region("notes_column", col_bbox, None, 2448.0, 1584.0)
    det_id = tab._store.save_detection(
        doc_id=tab._doc_id,
        page=tab._page,
        run_id=tab._run_id or "manual",
        element_type="notes_column",
        bbox=col_bbox,
        text_content=text_content,
        features=features,
    )
    tab._store.save_correction(
        doc_id=tab._doc_id,
        page=tab._page,
        correction_type="add",
        corrected_label="notes_column",
        corrected_bbox=col_bbox,
        detection_id=det_id,
        session_id=tab._session_id,
    )

    col_box = CanvasBox(
        detection_id=det_id,
        element_type="notes_column",
        confidence=None,
        text_content=text_content,
        features=features,
        pdf_bbox=col_bbox,
        corrected=True,
    )
    tab._canvas_boxes.append(col_box)
    tab._draw_box(col_box)

    if col_box.rect_id:
        tab._canvas.tag_lower(col_box.rect_id, "det_box")

    group_id = tab._store.create_group(
        doc_id=tab._doc_id,
        page=tab._page,
        group_label="notes_column",
        root_detection_id=det_id,
    )
    col_box.group_id = group_id
    col_box.is_group_root = True
    tab._groups[group_id] = {
        "label": "notes_column",
        "root_detection_id": det_id,
        "members": [det_id],
    }

    targets.sort(key=lambda cb: cb.pdf_bbox[1])
    grp = tab._groups[group_id]
    for i, cb in enumerate(targets, start=1):
        tab._store.add_to_group(group_id, cb.detection_id, sort_order=i)
        cb.group_id = group_id
        cb.is_group_root = False
        grp["members"].append(cb.detection_id)
        tab._draw_box(cb)

    tab._draw_group_links(group_id)

    tab._session_count += 1
    tab._update_session_label()
    tab._update_page_summary()
    tab._clear_multi_select()
    tab._select_box(col_box)
    n_children = len(targets)
    tab._status.configure(text=f"Created notes_column from {n_children} boxes")
