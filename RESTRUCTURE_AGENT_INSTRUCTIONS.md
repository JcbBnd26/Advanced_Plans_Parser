# RESTRUCTURE INSTRUCTIONS — VS Code Agent

## Overview

Reorganize `src/plancheck/` from a flat module layout into stage-based folders that mirror
the 5-stage pipeline: **ingest → tocr → vocrpp → vocr → reconcile**, plus post-pipeline
concerns (grouping, analysis, checks, export). Also reorganize `scripts/` and `tests/`.

**CRITICAL RULES:**
- NO logic changes. This is purely moving code and updating imports.
- After EVERY section below, run `pytest tests/ -x -q` and confirm all tests pass.
- If tests fail, fix the imports — never change logic.
- Preserve the top-level `plancheck/__init__.py` public API by re-exporting from new locations.

---

## SECTION 1: Create Folder Structure

Create these directories with empty `__init__.py` files:

```bash
mkdir -p src/plancheck/ingest
mkdir -p src/plancheck/tocr
mkdir -p src/plancheck/vocrpp
mkdir -p src/plancheck/vocr
mkdir -p src/plancheck/reconcile
mkdir -p src/plancheck/grouping
mkdir -p src/plancheck/analysis
mkdir -p src/plancheck/checks
mkdir -p src/plancheck/export

touch src/plancheck/ingest/__init__.py
touch src/plancheck/tocr/__init__.py
touch src/plancheck/vocrpp/__init__.py
touch src/plancheck/vocr/__init__.py
touch src/plancheck/reconcile/__init__.py
touch src/plancheck/grouping/__init__.py
touch src/plancheck/analysis/__init__.py
touch src/plancheck/checks/__init__.py
touch src/plancheck/export/__init__.py
```

Also create the scripts subfolders:

```bash
mkdir -p scripts/runners
mkdir -p scripts/gui
mkdir -p scripts/debug
mkdir -p scripts/overlays
mkdir -p scripts/diagnostics
mkdir -p scripts/utils
```

Run tests. They should still pass (nothing moved yet).

---

## SECTION 2: Move Leaf Modules (No Internal Dependents)

These files are imported by others but don't import other plancheck modules (or only import
`config.py` / `models.py`). Move them first since nothing breaks.

### 2A: vocrpp

```
src/plancheck/ocr_preprocess_pipeline.py → src/plancheck/vocrpp/preprocess.py
```

Update `src/plancheck/vocrpp/__init__.py`:
```python
from .preprocess import OcrPreprocessConfig, OcrPreprocessResult, preprocess_image_for_ocr

__all__ = ["OcrPreprocessConfig", "OcrPreprocessResult", "preprocess_image_for_ocr"]
```

Add backward-compat re-export — create `src/plancheck/ocr_preprocess_pipeline.py`:
```python
"""Backward compatibility — imports moved to plancheck.vocrpp.preprocess."""
from .vocrpp.preprocess import OcrPreprocessConfig, OcrPreprocessResult, preprocess_image_for_ocr  # noqa: F401
```

### 2B: checks

```
src/plancheck/semantic_checks.py → src/plancheck/checks/semantic_checks.py
```

Update `src/plancheck/checks/__init__.py`:
```python
from .semantic_checks import run_all_checks  # noqa: F401
```

Add backward-compat shim — create `src/plancheck/semantic_checks.py`:
```python
"""Backward compatibility — imports moved to plancheck.checks.semantic_checks."""
from .checks.semantic_checks import *  # noqa: F401,F403
```

### 2C: font_metrics

```
src/plancheck/font_metrics.py → src/plancheck/grouping/font_metrics.py
```

`font_metrics.py` has ZERO internal plancheck imports, so this is a clean move.

Add backward-compat shim — create `src/plancheck/font_metrics.py`:
```python
"""Backward compatibility — imports moved to plancheck.grouping.font_metrics."""
from .grouping.font_metrics import *  # noqa: F401,F403
```

### 2D: _graphics.py

```
src/plancheck/_graphics.py → src/plancheck/analysis/graphics.py
```

Internal imports to update in the moved file:
- `from .models import GraphicElement` → `from ..models import GraphicElement`

Add backward-compat shim — create `src/plancheck/_graphics.py`:
```python
"""Backward compatibility — imports moved to plancheck.analysis.graphics."""
from .analysis.graphics import *  # noqa: F401,F403
```

### 2E: _region_helpers.py

```
src/plancheck/_region_helpers.py → src/plancheck/analysis/region_helpers.py
```

Internal imports to update in the moved file:
- `from .models import BlockCluster, GlyphBox, GraphicElement` → `from ..models import BlockCluster, GlyphBox, GraphicElement`

Add backward-compat shim — create `src/plancheck/_region_helpers.py`:
```python
"""Backward compatibility — imports moved to plancheck.analysis.region_helpers."""
from .analysis.region_helpers import *  # noqa: F401,F403
```

### 2F: _ocr_engine.py

```
src/plancheck/_ocr_engine.py → src/plancheck/vocr/engine.py
```

This file has no internal plancheck imports. Clean move.

Add backward-compat shim — create `src/plancheck/_ocr_engine.py`:
```python
"""Backward compatibility — imports moved to plancheck.vocr.engine."""
from .vocr.engine import *  # noqa: F401,F403
```

**Run tests. Fix any import issues. All tests must pass before continuing.**

---

## SECTION 3: Move Analysis/Detection Modules

These modules import from `_region_helpers`, `config`, and `models`. Update their internal
imports to use the new paths.

### 3A: _abbreviation_detect.py

```
src/plancheck/_abbreviation_detect.py → src/plancheck/analysis/abbreviations.py
```

Update imports in the moved file:
```python
# OLD:
from ._region_helpers import _find_enclosing_rect, _find_text_blocks_in_region
from .config import GroupingConfig
from .models import AbbreviationEntry, AbbreviationRegion, BlockCluster, GraphicElement

# NEW:
from .region_helpers import _find_enclosing_rect, _find_text_blocks_in_region
from ..config import GroupingConfig
from ..models import AbbreviationEntry, AbbreviationRegion, BlockCluster, GraphicElement
```

Add backward-compat shim at old location:
```python
"""Backward compatibility — imports moved to plancheck.analysis.abbreviations."""
from .analysis.abbreviations import *  # noqa: F401,F403
```

### 3B: _revision_detect.py

```
src/plancheck/_revision_detect.py → src/plancheck/analysis/revisions.py
```

Update imports in the moved file:
```python
# OLD:
from ._region_helpers import _find_enclosing_rect, _find_text_blocks_in_region
from .config import GroupingConfig
from .models import (...)

# NEW:
from .region_helpers import _find_enclosing_rect, _find_text_blocks_in_region
from ..config import GroupingConfig
from ..models import (...)
```

Add backward-compat shim at old location.

### 3C: _standard_detail_detect.py

```
src/plancheck/_standard_detail_detect.py → src/plancheck/analysis/standard_details.py
```

Update imports in the moved file:
```python
# OLD:
from ._region_helpers import (...)
from .config import GroupingConfig
from .models import (...)

# NEW:
from .region_helpers import (...)
from ..config import GroupingConfig
from ..models import (...)
```

Add backward-compat shim at old location.

### 3D: _misc_title_detect.py

```
src/plancheck/_misc_title_detect.py → src/plancheck/analysis/misc_titles.py
```

Update imports in the moved file:
```python
# OLD:
from ._region_helpers import _bboxes_overlap
from .config import GroupingConfig
from .models import BlockCluster, GraphicElement, MiscTitleRegion

# NEW:
from .region_helpers import _bboxes_overlap
from ..config import GroupingConfig
from ..models import BlockCluster, GraphicElement, MiscTitleRegion
```

Add backward-compat shim at old location.

### 3E: _structural_boxes.py

```
src/plancheck/_structural_boxes.py → src/plancheck/analysis/structural_boxes.py
```

Update imports in the moved file:
```python
# OLD:
from .models import BlockCluster, GlyphBox, GraphicElement

# NEW:
from ..models import BlockCluster, GlyphBox, GraphicElement
```

Add backward-compat shim at old location:
```python
"""Backward compatibility — imports moved to plancheck.analysis.structural_boxes."""
from .analysis.structural_boxes import *  # noqa: F401,F403
```

### 3F: legends.py

```
src/plancheck/legends.py → src/plancheck/analysis/legends.py
```

This file has complex imports — it re-exports from other detection modules. Update ALL of them:

```python
# OLD:
from ._abbreviation_detect import detect_abbreviation_regions
from ._graphics import extract_graphics
from ._misc_title_detect import detect_misc_title_regions
from ._region_helpers import filter_graphics_outside_regions
from ._region_helpers import (...)
from ._revision_detect import detect_revision_regions
from ._standard_detail_detect import detect_standard_detail_regions
from .config import GroupingConfig
from .models import BlockCluster, GraphicElement, LegendEntry, LegendRegion

# NEW:
from .abbreviations import detect_abbreviation_regions
from .graphics import extract_graphics
from .misc_titles import detect_misc_title_regions
from .region_helpers import filter_graphics_outside_regions
from .region_helpers import (...)
from .revisions import detect_revision_regions
from .standard_details import detect_standard_detail_regions
from ..config import GroupingConfig
from ..models import BlockCluster, GraphicElement, LegendEntry, LegendRegion
```

Add backward-compat shim at old location.

### 3G: zoning.py

```
src/plancheck/zoning.py → src/plancheck/analysis/zoning.py
```

Update imports in the moved file:
```python
# OLD:
from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, NotesColumn

# NEW:
from ..config import GroupingConfig
from ..models import BlockCluster, GlyphBox, NotesColumn
```

Add backward-compat shim at old location.

### 3H: Update analysis/__init__.py

```python
from .abbreviations import detect_abbreviation_regions
from .graphics import extract_graphics
from .legends import detect_legend_regions
from .misc_titles import detect_misc_title_regions
from .region_helpers import (
    _bboxes_overlap,
    _find_enclosing_rect,
    _find_text_blocks_in_region,
    filter_graphics_outside_regions,
)
from .revisions import detect_revision_regions
from .standard_details import detect_standard_detail_regions
from .structural_boxes import (
    BoxType,
    SemanticRegion,
    StructuralBox,
    classify_structural_boxes,
    create_synthetic_regions,
    detect_semantic_regions,
    detect_structural_boxes,
    mask_blocks_by_structural_boxes,
)
from .zoning import PageZone, ZoneTag, classify_blocks, detect_zones, zone_summary
```

**Run tests. Fix any import issues. All tests must pass before continuing.**

---

## SECTION 4: Move Export Modules

### 4A: export.py

```
src/plancheck/export.py → src/plancheck/export/csv_export.py
```

This file has NO internal plancheck imports (it only uses stdlib). Clean move.

Add backward-compat shim at old location:
```python
"""Backward compatibility — imports moved to plancheck.export.csv_export."""
from .export.csv_export import *  # noqa: F401,F403
```

### 4B: overlay.py

```
src/plancheck/overlay.py → src/plancheck/export/overlay.py
```

Update imports in the moved file:
```python
# OLD:
from .config import GroupingConfig
from .models import (...)

# NEW:
from ..config import GroupingConfig
from ..models import (...)
```

Add backward-compat shim at old location.

### 4C: page_data.py

```
src/plancheck/page_data.py → src/plancheck/export/page_data.py
```

Update imports in the moved file:
```python
# OLD:
from plancheck.models import BlockCluster, GlyphBox, NotesColumn

# NEW:
from ..models import BlockCluster, GlyphBox, NotesColumn
```

Add backward-compat shim at old location.

### 4D: Update export/__init__.py

```python
from .csv_export import (
    export_abbreviations_csv,
    export_blocks_csv,
    export_from_manifest,
    export_legends_csv,
    export_notes_csv,
    export_page_results,
    export_page_summary_csv,
    export_revisions_csv,
    export_standard_details_csv,
)
from .overlay import draw_columns_overlay, draw_lines_overlay, draw_overlay
from .page_data import deserialize_page, serialize_page
```

**Run tests. Fix any import issues. All tests must pass before continuing.**

---

## SECTION 5: Move Preprocess (TOCR Support)

```
src/plancheck/preprocess.py → src/plancheck/tocr/preprocess.py
```

Update imports in the moved file:
```python
# OLD:
from .models import GlyphBox

# NEW:
from ..models import GlyphBox
```

Add backward-compat shim at old location:
```python
"""Backward compatibility — imports moved to plancheck.tocr.preprocess."""
from .tocr.preprocess import *  # noqa: F401,F403
```

**Run tests.**

---

## SECTION 6: Move Grouping Module

This is the biggest file (2000 lines). Move it whole first, then optionally split later.

```
src/plancheck/grouping.py → src/plancheck/grouping/clustering.py
```

Update imports in the moved file:
```python
# OLD:
from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, Line, NotesColumn, RowBand, Span

# NEW:
from ..config import GroupingConfig
from ..models import BlockCluster, GlyphBox, Line, NotesColumn, RowBand, Span
```

Update `src/plancheck/grouping/__init__.py`:
```python
"""Geometry-first clustering: rows → lines → blocks → notes columns."""
from .clustering import (
    build_clusters_v2,
    build_lines,
    compute_median_space_gap,
    group_blocks,
    group_blocks_from_lines,
    group_notes_columns,
    group_rows,
    link_continued_columns,
    mark_headers,
    mark_notes,
    mark_tables,
    split_line_spans,
    split_wide_lines,
)

__all__ = [
    "build_clusters_v2",
    "build_lines",
    "compute_median_space_gap",
    "group_blocks",
    "group_blocks_from_lines",
    "group_notes_columns",
    "group_rows",
    "link_continued_columns",
    "mark_headers",
    "mark_notes",
    "mark_tables",
    "split_line_spans",
    "split_wide_lines",
]
```

Add backward-compat shim — create `src/plancheck/grouping.py`:

**WAIT** — there's a naming conflict. The file `grouping.py` and the folder `grouping/` can't
coexist. Once the folder `grouping/` exists with `__init__.py`, Python will use the package.
The backward-compat approach here is: the `grouping/__init__.py` re-exports everything that
`grouping.py` used to export. Any code that does `from plancheck.grouping import X` will
still work because `plancheck.grouping` is now the package, and the `__init__.py` re-exports `X`.

**So no shim file is needed for `grouping.py`** — delete the original `src/plancheck/grouping.py`
after moving its contents to `src/plancheck/grouping/clustering.py`. The `__init__.py` handles
backward compat.

**Run tests.**

---

## SECTION 7: Split ocr_reconcile.py Between vocr/ and reconcile/

The file `src/plancheck/ocr_reconcile.py` contains two conceptually separate things:
1. `extract_vocr_tokens()` — belongs in `vocr/`
2. Everything else (reconcile logic, overlays) — belongs in `reconcile/`

### 7A: Extract vocr token extraction

Copy the following functions from `ocr_reconcile.py` into `src/plancheck/vocr/extract.py`:
- `extract_vocr_tokens()` (line ~1124)
- Any private helpers it calls: `_extract_ocr_tokens()` (line ~217), `_dedup_tiles()` (line ~185),
  `_ocr_one_tile()` (line ~125)

Update the imports in the new file to use `..config`, `..models`.

Update `src/plancheck/vocr/__init__.py`:
```python
from .extract import extract_vocr_tokens  # noqa: F401
```

### 7B: Move reconciliation logic

```
src/plancheck/ocr_reconcile.py → src/plancheck/reconcile/reconcile.py
```

**Remove** `extract_vocr_tokens`, `_extract_ocr_tokens`, `_dedup_tiles`, `_ocr_one_tile`
from this file (they now live in `vocr/extract.py`).

Update imports in the moved file:
```python
# OLD:
from .config import GroupingConfig
from .models import GlyphBox

# NEW:
from ..config import GroupingConfig
from ..models import GlyphBox
```

Move the overlay functions (`draw_reconcile_debug`, `draw_symbol_overlay`) into
`src/plancheck/export/reconcile_overlay.py` instead, since they are visualization code.
Update their imports to `from ..models import ...` and `from ..config import ...`.

Update `src/plancheck/reconcile/__init__.py`:
```python
from .reconcile import MatchRecord, ReconcileResult, SymbolCandidate, reconcile_ocr  # noqa: F401
```

### 7C: Backward-compat shim

Create `src/plancheck/ocr_reconcile.py`:
```python
"""Backward compatibility — imports split between plancheck.vocr and plancheck.reconcile."""
from .vocr.extract import extract_vocr_tokens  # noqa: F401
from .reconcile.reconcile import (  # noqa: F401
    MatchRecord,
    ReconcileResult,
    SymbolCandidate,
    reconcile_ocr,
)
from .export.reconcile_overlay import draw_reconcile_debug, draw_symbol_overlay  # noqa: F401
```

**Run tests.**

---

## SECTION 8: Update Top-Level __init__.py

Replace `src/plancheck/__init__.py` to import from the new locations. Keep the exact same
public API surface so nothing downstream breaks:

```python
"""Core geometry-first grouping package for plan checking."""

from .config import GroupingConfig
from .models import BlockCluster, GlyphBox, Line, RowBand, Span, SuspectRegion

from .grouping import (
    build_clusters_v2,
    build_lines,
    group_blocks,
    group_blocks_from_lines,
    group_rows,
    mark_notes,
    mark_tables,
)

try:
    from .reconcile import reconcile_ocr
    from .vocr import extract_vocr_tokens
    from .export.reconcile_overlay import draw_reconcile_debug, draw_symbol_overlay
except ImportError:
    reconcile_ocr = None
    extract_vocr_tokens = None
    draw_reconcile_debug = None
    draw_symbol_overlay = None

from .analysis.structural_boxes import (
    BoxType,
    SemanticRegion,
    StructuralBox,
    classify_structural_boxes,
    create_synthetic_regions,
    detect_semantic_regions,
    detect_structural_boxes,
    mask_blocks_by_structural_boxes,
)
from .export.csv_export import (
    export_abbreviations_csv,
    export_blocks_csv,
    export_from_manifest,
    export_legends_csv,
    export_notes_csv,
    export_page_results,
    export_page_summary_csv,
    export_revisions_csv,
    export_standard_details_csv,
)
from .vocrpp import (
    OcrPreprocessConfig,
    OcrPreprocessResult,
    preprocess_image_for_ocr,
)
from .export.overlay import draw_overlay
from .export.page_data import deserialize_page, serialize_page
from .pipeline import (
    STAGE_ORDER,
    SkipReason,
    StageResult,
    gate,
    input_fingerprint,
    run_stage,
)
from .tocr.preprocess import estimate_skew_degrees, nms_prune, rotate_boxes
from .analysis.zoning import PageZone, ZoneTag, classify_blocks, detect_zones, zone_summary

# Keep __all__ identical to the original
__all__ = [
    "GroupingConfig",
    "GlyphBox",
    "RowBand",
    "Line",
    "Span",
    "BlockCluster",
    "SuspectRegion",
    "nms_prune",
    "estimate_skew_degrees",
    "rotate_boxes",
    "draw_overlay",
    "serialize_page",
    "deserialize_page",
    "group_rows",
    "group_blocks",
    "group_blocks_from_lines",
    "build_lines",
    "mark_tables",
    "build_clusters_v2",
    "reconcile_ocr",
    "extract_vocr_tokens",
    "draw_reconcile_debug",
    "draw_symbol_overlay",
    "OcrPreprocessConfig",
    "OcrPreprocessResult",
    "preprocess_image_for_ocr",
    "STAGE_ORDER",
    "SkipReason",
    "StageResult",
    "gate",
    "run_stage",
    "input_fingerprint",
    "PageZone",
    "ZoneTag",
    "detect_zones",
    "classify_blocks",
    "zone_summary",
    "export_page_results",
    "export_page_summary_csv",
    "export_notes_csv",
    "export_abbreviations_csv",
    "export_legends_csv",
    "export_standard_details_csv",
    "export_revisions_csv",
    "export_blocks_csv",
    "export_from_manifest",
    "BoxType",
    "StructuralBox",
    "SemanticRegion",
    "detect_structural_boxes",
    "classify_structural_boxes",
    "create_synthetic_regions",
    "detect_semantic_regions",
    "mask_blocks_by_structural_boxes",
]
```

**Run tests.**

---

## SECTION 9: Move Scripts into Subfolders

Move scripts into organized subfolders. Update any cross-script imports
(e.g., `gui.py` imports `from run_pdf_batch import ...` and `from overlay_viewer import ...`).

```
scripts/run_pdf_batch.py    → scripts/runners/run_pdf_batch.py
scripts/run_pdf_page.py     → scripts/runners/run_pdf_page.py
scripts/run_from_args.py    → scripts/runners/run_from_args.py
scripts/new_run.py          → scripts/runners/new_run.py

scripts/gui.py              → scripts/gui/gui.py
scripts/overlay_viewer.py   → scripts/gui/overlay_viewer.py

scripts/_debug_char_gaps.py → scripts/debug/_debug_char_gaps.py
scripts/_debug_slash.py     → scripts/debug/_debug_slash.py
scripts/_debug_notes.py     → scripts/debug/_debug_notes.py
scripts/_debug_nc3.py       → scripts/debug/_debug_nc3.py

scripts/overlay_headers_red.py   → scripts/overlays/overlay_headers_red.py
scripts/overlay_notes_green.py   → scripts/overlays/overlay_notes_green.py
scripts/overlay_notes_purple.py  → scripts/overlays/overlay_notes_purple.py

scripts/run_benchmark.py               → scripts/diagnostics/run_benchmark.py
scripts/run_tuning_harness.py           → scripts/diagnostics/run_tuning_harness.py
scripts/run_font_metrics_diagnostics.py → scripts/diagnostics/run_font_metrics_diagnostics.py
scripts/run_ocr_preprocess.py           → scripts/diagnostics/run_ocr_preprocess.py
scripts/run_grouping.py                 → scripts/diagnostics/run_grouping.py

scripts/extract_page.py → scripts/utils/extract_page.py
scripts/tag_list.py     → scripts/utils/tag_list.py
```

**Key import fixes in scripts:**

In `scripts/gui/gui.py`:
```python
# OLD:
from overlay_viewer import OverlayViewerTab
from run_pdf_batch import cleanup_old_runs, run_pdf
from tag_list import TAG_DESCRIPTIONS, TAG_LIST

# NEW:
from scripts.gui.overlay_viewer import OverlayViewerTab
from scripts.runners.run_pdf_batch import cleanup_old_runs, run_pdf
from scripts.utils.tag_list import TAG_DESCRIPTIONS, TAG_LIST
```

NOTE: The scripts use relative imports and `sys.path` manipulation. You may need to adjust
the `sys.path.insert(0, ...)` lines to account for the new subfolder depth, or switch to
using `-m` style invocation. Check each script's `sys.path` setup.

In `scripts/diagnostics/run_benchmark.py` and `run_tuning_harness.py`:
```python
# OLD:
from run_pdf_batch import cleanup_old_runs, run_pdf

# NEW — adjust sys.path or use:
from scripts.runners.run_pdf_batch import cleanup_old_runs, run_pdf
```

Also update `launch_gui.bat` if it references `scripts/gui.py` to `scripts/gui/gui.py`.

**Run the GUI manually to confirm it still launches.**

---

## SECTION 10: Move Tests to Mirror Source Structure

```bash
mkdir -p tests/tocr
mkdir -p tests/vocrpp
mkdir -p tests/reconcile
mkdir -p tests/grouping
mkdir -p tests/analysis
mkdir -p tests/checks
mkdir -p tests/export
```

```
tests/test_preprocess.py              → tests/tocr/test_preprocess.py
tests/test_ocr_preprocess_pipeline.py → tests/vocrpp/test_ocr_preprocess_pipeline.py
tests/test_ocr_reconcile.py           → tests/reconcile/test_ocr_reconcile.py

tests/test_grouping.py                → tests/grouping/test_grouping.py
tests/test_font_metrics.py            → tests/grouping/test_font_metrics.py

tests/test_structural_boxes.py        → tests/analysis/test_structural_boxes.py
tests/test_legends.py                 → tests/analysis/test_legends.py
tests/test_abbreviation_detect.py     → tests/analysis/test_abbreviation_detect.py
tests/test_revision_detect.py         → tests/analysis/test_revision_detect.py
tests/test_standard_detail_detect.py  → tests/analysis/test_standard_detail_detect.py
tests/test_misc_title_detect.py       → tests/analysis/test_misc_title_detect.py
tests/test_region_helpers.py          → tests/analysis/test_region_helpers.py
tests/test_zoning.py                  → tests/analysis/test_zoning.py

tests/test_semantic_checks.py         → tests/checks/test_semantic_checks.py

tests/test_export.py                  → tests/export/test_export.py
tests/test_overlay.py                 → tests/export/test_overlay.py
```

Keep at root:
```
tests/conftest.py        (shared fixtures)
tests/test_models.py     (root-level models)
tests/test_config.py     (root-level config)
tests/test_pipeline.py   (root-level pipeline)
tests/test_integration.py (cross-cutting)
```

**Important:** Tests that import from `conftest` (e.g., `from conftest import make_box`) will
need adjustment. Either:
1. Add `conftest.py` to each subfolder that imports from it, OR
2. Change imports to `from tests.conftest import make_box` and ensure `tests/` is in the path, OR
3. Create a `tests/conftest.py` that pytest auto-discovers (it already is — subfolders will
   inherit it if you add empty `__init__.py` files or `conftest.py` files to subfolders).

The safest approach: add an empty `conftest.py` to each test subfolder that imports the shared
fixtures:

```python
# tests/grouping/conftest.py (and same for other subfolders)
from tests.conftest import *  # noqa: F401,F403
```

Or simply ensure pytest's `rootdir` is set correctly in `pytest.ini` — the existing `conftest.py`
at `tests/` root should be auto-discovered for all subfolders.

**Update test imports that use old module paths.** The backward-compat shims will handle most
of these, but it's cleaner to update them to the new paths:

| Old Import | New Import |
|------------|------------|
| `from plancheck._abbreviation_detect import ...` | `from plancheck.analysis.abbreviations import ...` |
| `from plancheck._revision_detect import ...` | `from plancheck.analysis.revisions import ...` |
| `from plancheck._standard_detail_detect import ...` | `from plancheck.analysis.standard_details import ...` |
| `from plancheck._misc_title_detect import ...` | `from plancheck.analysis.misc_titles import ...` |
| `from plancheck._structural_boxes import ...` | `from plancheck.analysis.structural_boxes import ...` |
| `from plancheck._region_helpers import ...` | `from plancheck.analysis.region_helpers import ...` |
| `from plancheck.ocr_reconcile import ...` | `from plancheck.reconcile import ...` |
| `from plancheck.ocr_preprocess_pipeline import ...` | `from plancheck.vocrpp import ...` |
| `from plancheck.preprocess import ...` | `from plancheck.tocr.preprocess import ...` |
| `from plancheck.semantic_checks import ...` | `from plancheck.checks.semantic_checks import ...` |
| `from plancheck.export import ...` | `from plancheck.export.csv_export import ...` |
| `from plancheck.overlay import ...` | `from plancheck.export.overlay import ...` |
| `from plancheck.legends import ...` | `from plancheck.analysis.legends import ...` |
| `from plancheck.zoning import ...` | `from plancheck.analysis.zoning import ...` |
| `from plancheck.font_metrics import ...` | `from plancheck.grouping.font_metrics import ...` |
| `from plancheck.page_data import ...` | `from plancheck.export.page_data import ...` |
| `from plancheck.grouping import ...` | `from plancheck.grouping import ...` (unchanged — __init__.py re-exports) |

**Run full test suite: `pytest tests/ -x -q`**

---

## SECTION 11: Clean Up

### 11A: Delete backup/patch folders

```bash
rm -rf backup_pre_tocr_patch/
rm -rf tocr_semantic_bbox_patch/
```

These are version control's job. Git history preserves them.

### 11B: Remove backward-compat shims (optional, can defer)

Once all tests pass and all scripts work, you can optionally remove the backward-compat
shim files at the old locations. This is a separate commit. Only do this after confirming
nothing external depends on the old paths.

The shim files to eventually remove:
```
src/plancheck/ocr_preprocess_pipeline.py  (shim)
src/plancheck/semantic_checks.py          (shim)
src/plancheck/font_metrics.py             (shim)
src/plancheck/_graphics.py                (shim)
src/plancheck/_region_helpers.py          (shim)
src/plancheck/_ocr_engine.py              (shim)
src/plancheck/_abbreviation_detect.py     (shim)
src/plancheck/_revision_detect.py         (shim)
src/plancheck/_standard_detail_detect.py  (shim)
src/plancheck/_misc_title_detect.py       (shim)
src/plancheck/_structural_boxes.py        (shim)
src/plancheck/legends.py                  (shim)
src/plancheck/zoning.py                   (shim)
src/plancheck/export.py                   (shim)
src/plancheck/overlay.py                  (shim)
src/plancheck/page_data.py               (shim)
src/plancheck/preprocess.py              (shim)
src/plancheck/ocr_reconcile.py           (shim)
```

### 11C: Final verification

```bash
pytest tests/ -v
```

All tests green. The restructure is complete.

---

## FINAL STRUCTURE

```
src/plancheck/
├── __init__.py              # Public API re-exports (unchanged surface)
├── config.py                # GroupingConfig
├── models.py                # All dataclasses
├── pipeline.py              # Stage orchestration
│
├── ingest/                  # Stage 1 (future: extract render_page_image here)
│   └── __init__.py
├── tocr/                    # Stage 2
│   ├── __init__.py
│   └── preprocess.py        # nms_prune, skew estimation, rotate_boxes
├── vocrpp/                  # Stage 3
│   ├── __init__.py
│   └── preprocess.py        # OcrPreprocessConfig, preprocess_image_for_ocr
├── vocr/                    # Stage 4
│   ├── __init__.py
│   ├── engine.py            # PaddleOCR wrapper
│   └── extract.py           # extract_vocr_tokens
├── reconcile/               # Stage 5
│   ├── __init__.py
│   └── reconcile.py         # reconcile_ocr, symbol injection
│
├── grouping/                # Post-OCR clustering
│   ├── __init__.py
│   ├── clustering.py        # All grouping logic (2000 lines — split later)
│   └── font_metrics.py      # Font analysis
│
├── analysis/                # Structural + semantic detection
│   ├── __init__.py
│   ├── structural_boxes.py
│   ├── legends.py
│   ├── abbreviations.py
│   ├── revisions.py
│   ├── standard_details.py
│   ├── misc_titles.py
│   ├── region_helpers.py
│   ├── graphics.py
│   └── zoning.py
│
├── checks/                  # Semantic validation
│   ├── __init__.py
│   └── semantic_checks.py
│
└── export/                  # Output generation
    ├── __init__.py
    ├── csv_export.py
    ├── overlay.py
    ├── reconcile_overlay.py
    └── page_data.py

scripts/
├── runners/
├── gui/
├── debug/
├── overlays/
├── diagnostics/
└── utils/

tests/
├── conftest.py
├── test_models.py
├── test_config.py
├── test_pipeline.py
├── test_integration.py
├── tocr/
├── vocrpp/
├── reconcile/
├── grouping/
├── analysis/
├── checks/
└── export/
```
