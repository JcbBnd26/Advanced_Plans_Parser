# UI Feature Audit

Generated as part of the UI Restructure Plan — Phase 1.

---

## Tab 1: Pipeline (`tab_pipeline.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| Select File… button | Opens file dialog for PDF selection | business | PDF File section | Pipeline tab (keep) |
| File label | Shows selected PDF filename | business | PDF File section | Pipeline tab (keep) |
| Clear button | Clears selected PDF | business | PDF File section | Pipeline tab (keep) |
| All Pages radio | Select all pages for processing | business | Page Selection section | Pipeline tab (keep) |
| Single Page radio + entry | Process a single page | business | Page Selection section | Pipeline tab (keep) |
| Page Range radio + entries | Process a page range | business | Page Selection section | Pipeline tab (keep) |
| Load Config… button | Load YAML/TOML pipeline config | business | Config File section | Pipeline tab (keep) |
| Save Config… button | Save current config to file | business | Config File section | Pipeline tab (keep) |
| Config path label | Shows current config source | business | Config File section | Pipeline tab (keep) |
| TOCR checkbox | Enable pdfplumber text extraction | business | Pipeline Stages section | Pipeline tab (keep) |
| Render DPI spinbox | Set render resolution (72–300) | business | Pipeline Stages section | Pipeline tab (keep) |
| Deskew checkbox | Enable rotation correction | business | Pipeline Stages section | Pipeline tab (keep) |
| LLM Checks checkbox | Enable semantic analysis | business | Pipeline Stages section | Pipeline tab (keep) |
| Enable ML relabeling checkbox | Master ML toggle | business | ML Settings section | Pipeline tab (keep) |
| Hierarchical title refinement checkbox | Enable Stage 2 routing | business | ML Settings section | Pipeline tab (keep) |
| Stage 1 model path entry + Browse | Path to Stage 1 classifier | business | ML Settings section | Pipeline tab (keep) |
| Stage 2 model path entry + Browse | Path to Stage 2 classifier | business | ML Settings section | Pipeline tab (keep) |
| Relabel confidence threshold | ML confidence threshold | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Auto-retrain on startup checkbox | Retrain check at launch | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Retrain threshold entry | Corrections before retrain | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Enable feature cache checkbox | Reuse encoded features | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Enable drift detection checkbox | Compare to training stats | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Drift threshold entry | Feature shift trigger level | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Drift stats path entry + Browse | Drift statistics file | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Comparison F1 threshold entry | Per-class F1 delta | backend | Advanced ML Runtime | Settings → ML Runtime… |
| Enable vision features checkbox | Image feature extraction | backend | Optional ML Features | Settings → ML Features… |
| Vision backbone entry | Torchvision backbone name | backend | Optional ML Features | Settings → ML Features… |
| Enable text embeddings checkbox | Sentence embeddings | backend | Optional ML Features | Settings → ML Features… |
| Embeddings model entry | Sentence-transformer model | backend | Optional ML Features | Settings → ML Features… |
| Enable layout model checkbox | LayoutLMv3 layout prediction | backend | Optional ML Features | Settings → ML Features… |
| Layout model entry | LayoutLMv3 checkpoint path | backend | Optional ML Features | Settings → ML Features… |
| Enable GNN post-processing checkbox | GNN relational classification | backend | Optional ML Features | Settings → ML Features… |
| GNN model path entry + Browse | GNN checkpoint path | backend | Optional ML Features | Settings → ML Features… |
| GNN hidden dim entry | GNN feature width | backend | Optional ML Features | Settings → ML Features… |
| GNN early-stop patience entry | GNN training patience | backend | Optional ML Features | Settings → ML Features… |
| LLM provider entry | Provider key (ollama/openai/anthropic) | backend | LLM Runtime | Settings → LLM Configuration… |
| LLM model entry | Model name for provider | backend | LLM Runtime | Settings → LLM Configuration… |
| LLM API base entry | Provider endpoint URL | backend | LLM Runtime | Settings → LLM Configuration… |
| LLM policy entry | Runtime policy | backend | LLM Runtime | Settings → LLM Configuration… |
| LLM temperature entry | Sampling temperature | backend | LLM Runtime | Settings → LLM Configuration… |
| LLM API key entry (masked) | API key for hosted providers | backend | LLM Runtime | Settings → LLM Configuration… |
| Run Processing button | Start pipeline execution | business | Button row | Pipeline tab (keep) |
| ABORT button | Cancel running pipeline | business | Button row | Pipeline tab (keep) |
| ErrorPanel | Navigable error display | business | Below buttons | Pipeline tab (keep) |
| StageProgressBar | Horizontal stage indicator with timers | business | Below error panel | Pipeline tab (keep) |
| Copy times button | Copy stage times to clipboard | business | Next to stage bar | Pipeline tab (keep) |
| LogPanel | Scrollable color-coded console | business | Bottom of tab | Pipeline tab (keep) |

---

## Tab 2: Runs & Reports (`tab_runs.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| RunBrowserWidget | Treeview of past runs with filtering | business | Left panel | Keep |
| ReportViewerWidget | Manifest details, artifact export | business | Right panel | Keep |

---

## Tab 3: Database (`tab_database.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| Document tree | Treeview of filenames/pages | backend | Left panel | Keep (hide via View menu) |
| Detail panel | Scrollable detection detail | backend | Right panel | Keep (hide via View menu) |
| Run filter combobox | Filter by run | backend | Toolbar | Keep (hide via View menu) |
| Refresh button | Reload database | backend | Toolbar | Keep (hide via View menu) |
| Snapshot DB button | Create DB snapshot | backend | Toolbar | Keep (hide via View menu) |
| Restore button | Restore DB from snapshot | backend | Toolbar | Keep (hide via View menu) |

---

## Tab 4: Diagnostics (`tab_diagnostics.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| FontDiagnosticsSection | Font metric analysis | backend | Diagnostics sections | Keep (hide via View menu) |
| BenchmarkSection | A/B/C/D benchmark runner | backend | Diagnostics sections | Keep (hide via View menu) |
| MLCalibrationSection | ML calibration tools | backend | Diagnostics sections | Keep (hide via View menu) |
| MLRuntimeSummarySection | ML runtime stats | backend | Diagnostics sections | Keep (hide via View menu) |
| TrainingProgressSection | Training progress charts | backend | Diagnostics sections | Keep (hide via View menu) |
| ModelComparisonSection | Model A/B comparison | backend | Diagnostics sections | Keep (hide via View menu) |
| LayoutModelSection | LayoutLMv3 tools | backend | Diagnostics sections | Keep (hide via View menu) |
| TextEmbeddingsSection | Embedding diagnostics | backend | Diagnostics sections | Keep (hide via View menu) |
| LLMSemanticChecksSection | LLM check diagnostics | backend | Diagnostics sections | Keep (hide via View menu) |
| CrossPageGNNSection | Cross-page GNN tools | backend | Diagnostics sections | Keep (hide via View menu) |
| LogPanel | Diagnostic log output | backend | Bottom of tab | Keep (hide via View menu) |

---

## Tab 5: Sheet Recreation (`tab_recreation.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| Run directory selector | Choose run for recreation | business | Settings area | Keep |
| Source PDF selector | Original PDF for overlays | business | Settings area | Keep |
| Recreation options | Page size, margins, etc. | business | Settings area | Keep |
| LogPanel | Recreation log output | business | Bottom of tab | Keep |

---

## Tab 6: ML Trainer (`tab_annotation.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| Canvas + bbox rendering | Interactive detection editor | business | Center panel | Keep (unchanged) |
| Page navigation | Prev/Next/Jump page controls | business | Top bar | Keep (unchanged) |
| Label corrections | Relabel dropdown + confirm | business | Right panel | Keep (unchanged) |
| Parent-child linking | Hierarchical structure editor | business | Right panel | Keep (unchanged) |
| Filter controls | Label/correction/confidence filters | business | Left panel | Keep (unchanged) |
| Context menu | Right-click actions | business | Overlay | Keep (unchanged) |
| Undo/Redo system | Correction history stack | business | Internal (Ctrl+Z/Y) | Wire to Edit menu |

---

## Tab 7: Query (`tab_query.py`)

| Widget | Description | Category | Current Location | New Home |
|--------|-------------|----------|------------------|----------|
| Run selector | Choose indexed run | business | Top bar | Keep |
| Chat transcript | Scrolling conversation history | business | Center | Keep |
| Query input | Text entry for questions | business | Bottom | Keep |
| Semantic search toggle | Search-only vs LLM mode | business | Options | Keep |
| Page/region filters | Constrain search scope | business | Options | Keep |
| Export button | Save chat to Markdown | business | Toolbar | Keep |
| Cost/status indicators | Token usage, model status | business | Status area | Keep |

---

## Cross-Tab Dependencies

| Source | Target | Event | Description |
|--------|--------|-------|-------------|
| Pipeline tab | Database tab | `pipeline_starting` | Database closes DB to avoid lock |
| Pipeline tab | Annotation tab | `pipeline_starting` | Annotation closes DB to avoid lock |
| Pipeline tab | All tabs | `pdf_changed` | New PDF selected |
| Pipeline tab | All tabs | `run_completed` | Pipeline finished |
| Runs tab | Pipeline tab | `load_config` | Load config from a past run |
| File menu | All tabs | `project_changed` | Project activated/switched |
| Settings dialogs (new) | Pipeline tab | `ml_config_changed` | ML runtime settings changed |
| Settings dialogs (new) | Pipeline tab | `llm_config_changed` | LLM settings changed |
| Settings dialogs (new) | Pipeline tab | `ml_features_changed` | ML features toggled |

---

## Future / Plan-Grading Features

No plan-grading features are currently live in the UI. All tabs serve either
business or backend purposes. The "future" category from the plan does not
apply to any existing widget.
