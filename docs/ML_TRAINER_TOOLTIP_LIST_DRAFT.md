## ML Trainer Tooltip List (Draft)

This is a draft list of suggested cursor-hover descriptions for the `ML Trainer` tab.

The goal is to give each important control a short explanation that helps a user understand:

- what the control does
- when to use it
- what effect it has on review or training

These are written as proposed tooltip strings, not implementation notes.

## Top Bar

### PDF Label

- `Currently loaded PDF for annotation and review.`

### Browse Button

- `Open a PDF file in the ML Trainer tab. Use this when you want to review detections or corrections directly.`

### Page Number Spinner

- `Current page number. Press Enter after changing it to jump to that page.`

### Page Count Label

- `Total number of pages in the loaded PDF.`

### Previous Page Button

- `Go to the previous page. Shortcut: Ctrl+Left Arrow.`

### Next Page Button

- `Go to the next page. Shortcut: Ctrl+Right Arrow.`

### DPI Spinner

- `Rendering resolution for the page view. Higher DPI improves visual detail but may load more slowly.`

### Zoom Out Button

- `Zoom out to see more of the page at once. Shortcut: -`

### Zoom In Button

- `Zoom in for closer inspection of boxes, text, and page details. Shortcut: +`

### Fit Button

- `Fit the full page into the current view. Shortcut: F.`

### Words Toggle

- `Show or hide word-level boxes detected from the PDF text layer. Use this when checking text extraction or selecting words for merges.`

## Inspector Basics

### Detection ID

- `Unique identifier for the selected detection. Useful when cross-checking saved corrections or debugging a specific item.`

### Type Selector

- `Element type for the selected detection. Change this when the current label is wrong or when adding a new detection.`

### Add Type Button

- `Register a new element type so it can be used in this session and future reviews.`

### Subtype Selector

- `Optional Stage 2 title subtype. Use this for title-related elements when a more specific label is needed.`

### Confidence Label

- `Confidence score for the selected detection. Lower values usually deserve closer review.`

### Text Box

- `Extracted text for the selected detection. Review this when checking OCR quality or deciding how to relabel an item.`

### Rescan Button

- `Re-extract text from the current PDF region. Use this after moving or resizing a box.`

## Main Review Actions

### Accept Button

- `Confirm that the selected detection is correct. Shortcut: A.`

### Relabel Button

- `Save the selected type as a correction for this detection. Use this when the box is correct but the label is wrong. Shortcut: R.`

### Reject Button

- `Mark the selected detection as a false positive. Shortcut: D.`

### Merge Button

- `Combine selected boxes or words into one detection. Use this when the pipeline split a single logical item into multiple pieces. Shortcut: M.`

### Multi-Selection Status

- `Shows how many boxes are selected together. Use multi-selection for merges, grouping, and batch review actions.`

## Model Suggestion Area

### Suggestion Label

- `Model suggestion for the selected detection, including confidence and whether it came from Stage 1 or Stage 2.`

### Suggestion Detail Label

- `Extra routing detail for the model suggestion. This may explain Stage 1 to Stage 2 refinement, low-confidence subtype results, or why Stage 2 was skipped.`

### Apply Suggestion Button

- `Apply the model's suggested label to the selected detection. Use this when the suggested type is correct and you want to save time.`

## Group Controls

### Group Label

- `Shows whether the selected detection belongs to a group and how it fits into that group.`

### Create Group Button

- `Create a new group using the selected box as the parent item. Shortcut: G.`

### Add To Group Button

- `Add the current selection to the active group. Use this for related boxes that belong under one parent item.`

### Remove From Group Button

- `Remove the selected box from its group. If the selected box is the parent, the group may be removed.`

## Mode Controls

### Select / Edit Mode

- `Use this mode to inspect, move, resize, relabel, merge, or group existing detections.`

### Add New Element Mode

- `Use this mode to draw a new detection box on the page when something important was missed.`

## Filter Controls

### Show All Button

- `Turn all element-type filters back on so every detection type becomes visible again.`

### Hide All Button

- `Turn all element-type filters off. Use this when you want to re-enable only one or two types for focused review.`

### Pick Color Button

- `Choose the display color for the currently selected filter type.`

### Element-Type Filter Checkboxes

- `Show or hide one detection type in the page view. Use these filters to focus on a specific class of elements.`

### Min Confidence Slider

- `Hide detections below the selected confidence threshold. Use this to focus on stronger predictions or isolate weaker ones for review.`

### Uncorrected Only Toggle

- `Show only detections that have not been corrected yet. Use this to focus on unfinished review work.`

## Session And Page Summary

### Session Label

- `Number of corrections saved during the current session.`

### Page Elements Summary

- `Shows how many detections of each type are on the current page. Use this for a quick sense of page content and coverage.`

## Model And Training Section

### Train Model Button

- `Retrain the main model from accepted and relabeled corrections. Use this after enough new corrections have accumulated.`

### Bootstrap Button

- `Create starter training data from stronger existing detections. Use this when you do not yet have enough manual corrections for a normal retrain.`

### Metrics Button

- `Open the latest training metrics, including accuracy and class-level performance.`

### History Button

- `Show past training runs so you can compare how the model has changed over time.`

### Importance Button

- `Show which input features influenced the trained model most strongly.`

### Model Status Label

- `Current model status, including whether a model is loaded and whether retraining is approaching or past the threshold.`

### Drift Indicator

- `Warns when the current page looks different from the data used to build the drift reference. Treat predictions more cautiously when drift is active.`

### Runtime Summary Label

- `Quick summary of the current ML session: routing mode, drift posture, and retrain readiness.`

## Stats Section

### Stats Label

- `Summary of documents, detections, corrections, training examples, and retrain readiness.`

### Refresh Stats Button

- `Recalculate annotation and training statistics from the correction database.`

### Clear Old Runs Button

- `Remove saved detection data from older pipeline runs while keeping corrections and training data.`

## Active Learning And Recovery

### Suggest Next Page Button

- `Jump to the page that is most likely to benefit from more annotation work.`

### Snapshot Button

- `Save a timestamped backup of the correction database before major edits or retraining.`

### Restore Button

- `Restore corrections from a previous snapshot if you need to recover older annotation work.`

## Footer

### Status Bar

- `Current status and latest completed action in the ML Trainer tab.`

### Progress Bar

- `Shows background progress for longer operations such as loading, processing, or training.`

## Suggested Tone Rules

Use these rules when implementing the final tooltips:

- keep the first sentence short and direct
- explain the action before the theory
- mention a common use case when helpful
- include shortcuts only when they save time or reduce confusion
- avoid internal terms unless the user already sees them in the UI
- keep most tooltips to one or two sentences

## Items That Deserve Special Care

These newer ML-related controls should get especially clear hover text:

- suggestion detail label
- subtype selector
- model status label
- drift indicator
- runtime summary label
- Train Model
- Bootstrap
- Suggest Next Page

These are the places where users are most likely to wonder what the ML system is doing and when they should trust or retrain it.