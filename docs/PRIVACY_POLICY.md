# Data-Privacy Policy for LLM Features

## Overview

Advanced Plan Parser includes optional LLM-powered features (query engine,
compliance assistant, entity extraction).  Because construction plans often
contain **proprietary**, **confidential**, or **personally-identifiable**
information we enforce a data-privacy policy that controls whether plan data
may leave the local machine.

## Policy Levels

| Value                  | Behaviour |
|------------------------|-----------|
| `local_only` (default) | **All LLM calls stay on-device.**  Only the `ollama` provider is permitted. Any attempt to call a cloud provider (`openai`, `anthropic`) raises an error immediately. |
| `cloud_allowed`        | Cloud LLM providers are permitted.  The caller is assumed to have already obtained any required consent. |
| `cloud_with_consent`   | Cloud providers are permitted **after** the user has given one-time consent through the GUI consent dialog or CLI prompt.  The first cloud call triggers a consent gate; subsequent calls proceed without re-prompting. |

## Configuration

Set the policy in your `GroupingConfig` or YAML/JSON config file:

```yaml
llm_policy: local_only     # default – no data leaves your machine
# llm_policy: cloud_allowed
# llm_policy: cloud_with_consent
```

Or pass it directly when creating an `LLMClient`:

```python
from plancheck.checks.llm_checks import LLMClient

client = LLMClient(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...",
    policy="cloud_allowed",   # explicitly allow cloud calls
)
```

## Enforcement Points

1. **`LLMClient.chat()`** — Every outgoing LLM call passes through
   `_enforce_policy()` before contacting on the provider.  A cloud provider +
   `local_only` policy raises `RuntimeError`.

2. **`run_llm_checks()`** — The pipeline's semantic-check stage receives the
   policy from `GroupingConfig.llm_policy` and forwards it to `LLMClient`.

3. **Query Engine / Compliance Assistant / Entity Extraction** (Phase 1) —
   All new LLM-dependent components receive the active policy at construction
   and respect it for every call.

## What Data Is Sent

When a cloud provider *is* permitted, the following data may be transmitted to
the provider's API:

* Extracted OCR text from plan notes/columns.
* Structural metadata (boxes, coordinates) in summarized form.
* User-authored prompts / queries.

**No images or raw PDF bytes are sent to cloud LLMs.**

## Recommendations

| Scenario | Recommended Policy |
|----------|-------------------|
| Internal / air-gapped deployment | `local_only` |
| Enterprise with approved cloud vendor | `cloud_allowed` |
| Shared workstation / demo | `cloud_with_consent` |

## Changing the Default

The default (`local_only`) is intentionally conservative.  To change it
project-wide, edit `src/plancheck/config.py` → `GroupingConfig.llm_policy`.
