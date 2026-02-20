# Advanced Plan Parser - Tag Reference

Complete reference of all tags, labels, and region types used in the application.

---

## Data Model Tags

### GlyphBox.origin

| Value | Description |
|-------|-------------|
| `"text"` | Default - text extracted from PDF |
| `"header_candidate"` | Detected as potential header text |

### BlockCluster.label

| Value | Description |
|-------|-------------|
| `"note_column_header"` | Header block for a notes column |
| `"notes_block"` | Content block within a notes column |
| `None` | Unlabeled block |

### BlockCluster Flags

| Flag | Type | Description |
|------|------|-------------|
| `is_table` | bool | Block detected as a table |
| `is_notes` | bool | Block is part of notes |
| `is_header` | bool | Block is a header |

### Region.tag (Zoning)

| Value | Description |
|-------|-------------|
| `"page"` | Whole page region |

---

## Region Types (Model Classes)

### Core Grouping Models

| Class | Description |
|-------|-------------|
| `GlyphBox` | Individual word box (smallest unit) |
| `RowBand` | Horizontal row of glyph boxes |
| `BlockCluster` | Cluster of rows (paragraph/table) |
| `NotesColumn` | Header + associated notes blocks |

### Legend & Abbreviation Models

| Class | Description |
|-------|-------------|
| `LegendRegion` | Legend header + entries with symbols |
| `LegendEntry` | Symbol + description |
| `AbbreviationRegion` | Abbreviation header + entries |
| `AbbreviationEntry` | Code + meaning |

### Title Block Models

| Class | Description |
|-------|-------------|
| `RevisionRegion` | Revision header + entries |
| `RevisionEntry` | Number + description + date |
| `MiscTitleRegion` | Miscellaneous title box (e.g., "OKLAHOMA DEPARTMENT OF TRANSPORTATION") |

### Standard Details Models

| Class | Description |
|-------|-------------|
| `StandardDetailRegion` | Standard details header + subheader + entries |
| `StandardDetailEntry` | Sheet number + description |

### Graphics Models

| Class | Description |
|-------|-------------|
| `GraphicElement` | Line, rect, or curve extracted from PDF |

---

## Overlay Colors

Visual representation colors used in the overlay PNG output.

### Core Elements

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Glyph Boxes | Coral | (255, 127, 80) | 1 |
| Row Bands | Slate Blue | (106, 90, 205) | 2 |
| Regular Blocks | Crimson | (220, 20, 60) | 3 |
| Table Blocks | Amber (filled) | (255, 191, 0) | 3 |
| Header Blocks | Indigo | (75, 0, 130) | 3 |

### Notes

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Notes Columns | Forest Green | (34, 139, 34) | 4 |

### Legend

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Legend Region | Deep Sky Blue | (0, 191, 255) | 4 |
| Legend Header | Steel Blue | (70, 130, 180) | 3 |

### Abbreviations

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Abbreviation Region | Orange | (255, 165, 0) | 4 |
| Abbreviation Header | Blue | (0, 0, 255) | 3 |
| Abbreviation Code | Green | (0, 255, 0) | 2 |
| Abbreviation Meaning | Yellow | (255, 255, 0) | 2 |
| Abbreviation Connector | Light Purple | (200, 150, 255) | 2 |

### Revisions

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Revision Region | Cadet Blue | (95, 158, 160) | 4 |
| Revision Header | Dark Slate Gray | (47, 79, 79) | 3 |
| Revision Entry Row | Powder Blue | (176, 224, 230) | 2 |

### Misc Title

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Misc Title Region | Deep Pink | (255, 20, 147) | 3 |
| Misc Title Combined Text | Goldenrod | (218, 165, 32) | 2 |

### Standard Details

| Tag | Color Name | RGB | Width |
|-----|------------|-----|-------|
| Standard Detail Region | Purple | (128, 0, 128) | 4 |
| Standard Detail Header | Royal Blue | (65, 105, 225) | 3 |
| Standard Detail Subheader | Orange | (255, 165, 0) | 2 |
| Standard Detail Sheet Number | Cornflower Blue | (100, 149, 237) | 2 |
| Standard Detail Description | Dark Goldenrod | (184, 134, 11) | 2 |

---

## Model Hierarchy

```
GlyphBox (word)
    └── RowBand (line of words)
            └── BlockCluster (paragraph/table)
                    ├── NotesColumn (notes header + blocks)
                    ├── LegendRegion (legend header + entries)
                    ├── AbbreviationRegion (abbrev header + entries)
                    ├── RevisionRegion (revision header + entries)
                    ├── MiscTitleRegion (title box)
                    └── StandardDetailRegion (details header + entries)
```

---

## File Locations

| Model | Source File |
|-------|-------------|
| Core models | `src/plancheck/models.py` |
| Zoning | `src/plancheck/analysis/zoning.py` |
| Overlay rendering | `src/plancheck/export/overlay.py` |
| Grouping logic | `src/plancheck/grouping/clustering.py` |
| Legend detection | `src/plancheck/analysis/legends.py` |
| Abbreviation detection | `src/plancheck/analysis/abbreviations.py` |
| Revision detection | `src/plancheck/analysis/revisions.py` |
| Standard detail detection | `src/plancheck/analysis/standard_details.py` |
| Graphics extraction | `src/plancheck/analysis/graphics.py` |
| Structural boxes | `src/plancheck/analysis/structural_boxes.py` |
