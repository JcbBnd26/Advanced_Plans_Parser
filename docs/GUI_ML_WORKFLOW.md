## GUI ML Workflow Guide (Draft)

This guide explains the newer ML-related features in the desktop GUI and when to use them.

It is written for day-to-day users of the GUI. It focuses on practical use, not internal implementation details.

## Quick Start

For a normal review session, use this order:

1. Open the `Pipeline` tab and select a PDF.
2. Check the ML settings you plan to use.
3. Run the document or page range.
4. Open `ML Trainer` and review detections.
5. Accept, relabel, or reject items as needed.
6. Retrain when enough corrections have built up.
7. Use `Diagnostics` if you need to check model health, confidence quality, or runtime setup.

## Pipeline Tab

### What To Use It For

Use the `Pipeline` tab to:

- select the file and page range
- choose which OCR and candidate stages should run
- control the main ML options
- load or save a config
- catch setup problems before saving or running

### Main ML Controls

#### ML Enabled

Turns the main classifier on or off during processing.

Use it when:

- you want ML-assisted labeling during the run
- you want to compare ML behavior against rule-based behavior

#### Hierarchical Routing

Turns on the second title-refinement step after the main classifier.

Use it when:

- you want title-family detections split into more specific title types
- you have a Stage 2 model available and want subtype suggestions

If the Stage 2 model is missing, the system falls back to Stage 1 behavior and tells you so.

#### Stage 1 Model Path

Points to the main element-classifier checkpoint.

Use it when:

- you trained a new main model
- you want to switch between saved models

#### Stage 2 Model Path

Points to the title-subtype model.

Use it when:

- hierarchical routing is enabled
- you want title refinement instead of title-family-only output

#### Retrain On Startup

Lets the GUI check for enough new corrections and retrain automatically when it starts.

Use it when:

- you regularly work in the same correction database
- you want the model to stay fresh without manual retraining every session

#### Retrain Threshold

Controls how many new corrections are needed before retraining is recommended or triggered.

Use it when:

- you want more frequent retraining with smaller batches
- you want to wait for larger, more stable correction sets

### Optional ML Features

These controls are useful, but only when the required files and packages are in place.

#### Drift Detection

Checks whether current detections look meaningfully different from the data used to build the drift reference.

Use it when:

- you are reviewing documents that may differ from your recent training set
- you want an early signal that model quality may be slipping

#### Candidate ML

Uses a learned model during the `VOCR Candidates` stage.

Use it when:

- you are actively evaluating candidate-stage ML
- the candidate model file is present and current

#### Candidate GNN Prior

Adds a graph-based prior to candidate ranking.

Use it when:

- you are testing or using the graph-based candidate workflow
- the prior checkpoint is available

#### Layout And LLM Options

These are for optional runtime features.

Use them only when:

- you have the required model or service configured
- you are intentionally testing those capabilities

### Validation Messages

The Pipeline tab validates ML settings when you:

- import a config
- load a config file
- save a config file
- start a run

There are two kinds of messages:

- `Blocking issues`: the setup is incomplete enough that the feature should not run
- `Warnings`: the run can continue, but behavior will be limited or fallback-based

### Common Blocking Cases

- candidate ML is enabled but its model file is missing
- candidate GNN prior is enabled but its checkpoint is missing
- layout runtime is enabled with no checkpoint set
- hosted LLM checks are enabled without an API key

### Common Warning Cases

- Stage 1 model is missing, so the run falls back to rules
- Stage 2 model is missing, so title refinement stays at Stage 1 only
- drift is enabled but the drift stats file is missing
- layout is still pointing at a base checkpoint instead of a tuned one

### When To Start Here

Start in the `Pipeline` tab when you are:

- opening a new document
- switching models
- deciding whether to use Stage 2
- checking whether drift or candidate ML is ready
- loading a saved working config

## ML Trainer Tab

### What To Use It For

Use the `ML Trainer` tab to review detections and improve the model through corrections.

This is where you:

- inspect detections on the page
- read model suggestions
- accept, relabel, or reject items
- watch retrain readiness
- start training when enough corrections exist

### Reading Model Suggestions

When you click a detection, the inspector may show:

- the suggested label
- the confidence score
- whether it came from `Stage 1` or `Stage 2`
- how the label was routed
- low-confidence alternatives for title refinement
- a note when Stage 2 was unavailable and the result stayed at Stage 1

### What The Suggestion Labels Mean

`Stage 1`

The main classifier produced the suggestion.

`Stage 2`

The system refined a title-family result into a more specific title subtype.

`Stage 2, low confidence`

The subtype suggestion may still be helpful, but it should be reviewed carefully.

`Stage 2 skipped`

The item stayed at the Stage 1 result because subtype refinement was not available.

If the suggested label is different from the current label, the `Apply` button appears. If the label is the same, the GUI may still show routing detail so you can understand why the model made that choice.

### Status Areas In The Inspector

The ML Trainer inspector gives you three useful status views:

#### Model Status

Shows whether a model is loaded and whether retraining is approaching or past the threshold.

Use it when:

- you want a quick answer about whether the current model is usable
- you want to know whether enough corrections exist for retraining

#### Drift Indicator

Shows whether the current page appears drifted compared with the configured drift reference.

Use it when:

- you notice odd suggestions on a page
- you want to judge whether to trust predictions as much as usual

#### Runtime Summary

Shows routing mode, drift posture, and retrain readiness in one line.

Use it when:

- you want a quick summary without switching tabs
- you need to confirm whether the session is running in Stage 1-only or hierarchical mode

### Training Buttons

#### Train Model

Starts retraining from accepted and relabeled corrections.

Use it when:

- the retrain threshold has been reached
- you finished a correction batch and want updated suggestions

#### Bootstrap

Builds starter training data from stronger existing detections.

Use it when:

- you are starting with very little human correction data
- you need an initial model before a longer annotation pass

#### Metrics, History, Importance

Use these when you want to inspect the last training result, compare runs, or understand which inputs are influencing the model.

### When To Retrain

Retrain when:

- the threshold has been reached
- the model is making repeated mistakes you have already corrected several times
- drift warnings are becoming common
- title subtype behavior is weak but you now have better corrected examples

Avoid retraining after only a few scattered edits.

## Diagnostics Tab

### What To Use It For

Use `Diagnostics` when you want to inspect the ML setup without editing detections.

It is mainly for:

- checking what runtime state the GUI is using
- reviewing confidence quality
- comparing training results over time

### ML Runtime Summary

This section gives a quick read on:

- routing mode
- whether Stage 1 and Stage 2 models are available
- whether drift detection is ready
- whether retraining is recommended
- whether candidate ML is active and ready
- whether layout runtime is configured

Use it when you want to confirm that the session is using the ML setup you expect.

### ML Calibration

This section helps you check whether confidence scores are trustworthy.

Use `Stage 1` when:

- you want to review confidence quality for the main classifier

Use `Stage 2` when:

- you want to review confidence quality for title subtype refinement

Run calibration after retraining or when users report that confidence values feel misleading.

### Training Progress And Comparison

Use these sections when you want to confirm that a new training run actually improved results.

## Common Workflows

### Standard Review Session

1. Open the PDF in `Pipeline`.
2. Confirm the Stage 1 settings are correct.
3. Run the document or page range.
4. Open `ML Trainer`.
5. Accept clear correct results.
6. Relabel repeated mistakes.
7. Refresh stats and check whether retraining is now recommended.

### Title Subtype Review

1. Enable hierarchical routing.
2. Confirm the Stage 2 model path is valid.
3. Run the document.
4. Open title-family detections in `ML Trainer`.
5. Read the routing detail shown under the suggestion.
6. Correct subtype errors.
7. Retrain when enough corrected examples exist.

### Drift Review

1. Enable drift detection in `Pipeline`.
2. Confirm the drift stats path is correct.
3. Run a representative document.
4. Open `ML Trainer` and watch for page drift warnings.
5. Open `Diagnostics` and review `ML Runtime Summary`.
6. If drift appears often, trust predictions less and plan retraining.

### Candidate ML Evaluation

1. Enable `VOCR Candidates`.
2. Enable candidate ML.
3. Confirm the candidate model path is valid.
4. If using the GNN prior, confirm that checkpoint too.
5. Run only when validation shows no blocking issues.
6. Compare results with a run that does not use candidate ML.

## Practical Tips

### Treat Warnings As Useful Information

Warnings usually mean the system can still run, but not in the exact mode implied by the controls.

### Change One Thing At A Time

If you are testing new ML behavior, avoid enabling every optional feature at once. It becomes much harder to understand what improved or got worse.

### Keep Model Paths Current

If a model is retrained, moved, or replaced, update the path in the GUI config right away.

### Check Diagnostics Before Blaming Annotation

If suggestions look wrong, check:

1. the validation messages in `Pipeline`
2. the `ML Runtime Summary` in `Diagnostics`
3. the model status and drift indicator in `ML Trainer`
4. calibration output if confidence seems unreliable

## Limits Of This Draft

- this is still a draft guide
- screenshots are not included yet
- some optional ML features depend on extra files or packages that may not be present in every environment
- the guide assumes the current tab names and flow remain stable

## Good Next Improvements

The next documentation pass should add:

- screenshots for each tab
- a troubleshooting section for common validation messages
- a short decision guide for retraining
- a glossary for Stage 1, Stage 2, drift, calibration, and candidate ML