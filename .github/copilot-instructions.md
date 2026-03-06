# GitHub Copilot Instructions for Advanced_Plan_Parser

These rules apply to all code in this repository.

## Code Organization Rules

### Single Responsibility & File Size Limits

Each module should have one reason to change. If a file grows past **800 lines**, check whether it's doing multiple jobs:
- If it **is** doing multiple jobs → **split it** into focused modules
- If it's doing **one job** that's genuinely complex → **leave it alone**

When splitting:
1. Extract cohesive functionality into separate modules
2. Use clear, descriptive module names reflecting their single purpose
3. Maintain backward compatibility by re-exporting from the original module if needed

### Indicators a file may be doing multiple jobs:
- Multiple unrelated class definitions
- Functions that serve different domains (e.g., GUI logic mixed with data processing)
- Imports spanning many unrelated libraries
- Difficulty naming the module with a single clear purpose
