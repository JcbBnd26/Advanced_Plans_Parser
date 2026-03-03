# Bug Analysis Report: Advanced Plans Parser

## Project Overview
This is a geometry-first PDF plan-sheet analysis pipeline for automated plan checking. It extracts text, graphics, and structural regions from CAD-origin PDFs and runs semantic checks against engineering standards.

## Analysis Date
March 2, 2026

## Overall Code Quality
The codebase demonstrates **good engineering practices** with:
- Proper use of type hints
- Comprehensive error handling
- Parameterized SQL queries (no SQL injection risks)
- Proper resource management (context managers for file operations)
- Dataclass usage for clean data structures
- No bare `except:` clauses
- No mutable default arguments
- No syntax errors

## Identified Issues

### 1. **CRITICAL: Potential Race Condition in Database WAL Mode**
**Location:** `src/plancheck/corrections/store.py:183`
```python
self._conn.execute("PRAGMA journal_mode=WAL")
```

**Issue:** The code enables Write-Ahead Logging (WAL) mode for SQLite. While this improves performance, it can cause issues in multi-threaded or multi-process scenarios if not handled carefully. The database file may be accessed by multiple processes (GUI, CLI, batch runners) simultaneously.

**Risk Level:** Medium
**Impact:** Could lead to database corruption or locking issues in concurrent scenarios

**Why this matters:** If multiple instances of your application run simultaneously (e.g., GUI and batch processor), WAL mode without proper locking can cause:
- File locking conflicts
- Data inconsistency
- Database corruption

**Recommendation:** 
- Add connection pooling or file-based locking
- Document that only one process should access the database at a time
- Consider implementing a connection manager that enforces single-writer access


### 2. **POTENTIAL BUG: Unsafe Array Access Patterns**
**Location:** `src/plancheck/grouping/clustering.py:234`
```python
current_span_indices: List[int] = [sorted_indices[0]]
```

**Issue:** Assumes `sorted_indices` has at least one element without prior validation. If `sorted_indices` is empty, this will raise an `IndexError`.

**Risk Level:** Medium
**Impact:** Runtime crash if input data is malformed or empty

**Context:** While the calling code may ensure `sorted_indices` is non-empty, defensive programming would add a guard here.

**Recommendation:**
```python
if not sorted_indices:
    return []  # or handle appropriately
current_span_indices: List[int] = [sorted_indices[0]]
```


### 3. **POTENTIAL BUG: Missing Validation in Gap Sorting**
**Location:** `src/plancheck/grouping/clustering.py:555`
```python
_, left_box, right_box = gaps_sorted[0]
```

**Issue:** Assumes `gaps_sorted` has at least one element. If `gaps` is empty, `gaps_sorted` will be empty and this will raise an `IndexError`.

**Risk Level:** Medium  
**Impact:** Runtime crash with malformed or unusual PDF structures

**Why this could happen:** Edge cases with PDFs that have unusual layouts or no detectable gaps between elements.

**Recommendation:** Add validation before accessing `gaps_sorted[0]`


### 4. **POTENTIAL BUG: Unsafe List Access in Column Detection**
**Location:** `src/plancheck/grouping/clustering.py:836`
```python
x_clusters: List[List[Line]] = [[line_x0s[0][1]]]
```

**Issue:** Assumes `line_x0s` is non-empty without checking. Empty input would cause `IndexError`.

**Risk Level:** Medium
**Impact:** Runtime crash on edge-case PDFs with no detectable lines

**Recommendation:** Add guard check before accessing first element


### 5. **STYLE ISSUE: Non-Pythonic Empty List Checks**
**Multiple Locations:**
- `src/plancheck/tocr/extract.py:465`
- `src/plancheck/tocr/extract.py:607`
- `src/plancheck/analysis/revisions.py:73`

**Pattern:**
```python
if len(boxes) == 0:
    return []
```

**Issue:** While not a bug, this is non-Pythonic. Better style is:
```python
if not boxes:
    return []
```

**Risk Level:** Low (style issue)
**Impact:** Code readability

**Why it matters:** Pythonic code is more maintainable and follows community standards. The `len()` check also has a slight performance cost.


### 6. **DESIGN CONSIDERATION: Missing Input Validation**
**Location:** Throughout the codebase

**Issue:** Many functions accept complex input parameters (lists of GlyphBox, BlockCluster objects, etc.) without validating that required fields are present or that data is well-formed.

**Example:** Functions that access `.rows[0]` generally check for empty rows, but don't validate that individual boxes have required attributes like `text`, `x0`, `y0`, etc.

**Risk Level:** Low-Medium
**Impact:** Potential AttributeError if malformed objects are passed

**Why this matters:** While Python's duck typing is flexible, PDF parsing deals with untrusted input (user-uploaded PDFs). Malformed PDFs could create objects with missing attributes.

**Recommendation:** Consider adding validation decorators or using Pydantic for stricter data validation


### 7. **POTENTIAL PERFORMANCE ISSUE: Repeated Dictionary Creation**
**Location:** `src/plancheck/pipeline.py` lines 1138-1960

**Pattern:**
```python
_mean_conf = sum(pr.ocr_confs) / len(pr.ocr_confs)
```

**Issue:** Multiple places compute statistics by iterating over lists. While protected against division by zero, these operations could be optimized if performance becomes an issue with large documents.

**Risk Level:** Low
**Impact:** Performance degradation on very large PDFs (hundreds of pages)

**Recommendation:** Consider caching computed statistics if they're reused


### 8. **EDGE CASE: Font Inflation Calculation**
**Location:** `src/plancheck/grouping/font_metrics.py:615`
```python
inflation_factor = reported_width / visual_width
```

**Issue:** Protected by earlier checks, but `visual_width` could theoretically be very close to zero (though not exactly zero) for extremely narrow or corrupted glyphs.

**Risk Level:** Low
**Impact:** Extreme inflation factor values that might break downstream assumptions

**Why this matters:** Division by very small numbers produces very large results. If visual_width is 0.001 and reported_width is 10, inflation_factor becomes 10,000.

**Recommendation:** Add bounds checking for reasonable inflation_factor ranges


### 9. **MISSING ERROR CONTEXT: Generic Exception Handling**
**Location:** `src/plancheck/pipeline.py:2029-2038`
```python
except Exception as exc:
    log.error("run_document page %d failed: %s", pg, exc)
```

**Issue:** Catches all exceptions generically. While it does log them, the original traceback is not preserved.

**Risk Level:** Low
**Impact:** Harder debugging when issues occur in production

**Why this matters:** When debugging production issues, full tracebacks are invaluable. Currently, you only get the exception message.

**Recommendation:**
```python
except Exception as exc:
    log.error("run_document page %d failed: %s", pg, exc, exc_info=True)
```


### 10. **CONFIGURATION RISK: No Runtime Validation**
**Location:** `src/plancheck/config.py`

**Issue:** While the `GroupingConfig` class has extensive validation functions (`_check_range`, `_check_positive`, etc.), it's unclear if these are actually called when configs are loaded.

**Risk Level:** Low
**Impact:** Invalid configurations could cause runtime errors or incorrect behavior

**Recommendation:** Ensure validation functions are called in a `__post_init__` method or during configuration loading


## Summary Statistics
- **Critical Issues:** 1 (database concurrency)
- **Medium Priority Bugs:** 4 (unsafe array access patterns)
- **Low Priority Issues:** 5 (style, performance, error handling)
- **Total Issues Found:** 10

## Priority Recommendations

### Immediate (Fix Now)
1. Add database connection locking/pooling for WAL mode
2. Add guards for array access patterns in clustering.py

### Short Term (Next Sprint)
3. Add traceback logging to exception handlers
4. Add input validation for core data structures
5. Add bounds checking for calculated values like inflation_factor

### Long Term (Technical Debt)
6. Refactor empty list checks to Pythonic style
7. Add performance profiling for large documents
8. Consider stricter type validation (Pydantic models)

## Testing Recommendations

To catch these bugs before they hit production:

1. **Edge Case Testing:** Create test PDFs with:
   - Empty pages
   - Single-element layouts
   - Malformed font data
   - Concurrent database access scenarios

2. **Fuzzing:** Run the pipeline against randomly corrupted PDFs to find edge cases

3. **Load Testing:** Test with 100+ page documents to identify performance bottlenecks

4. **Concurrency Testing:** Run multiple instances simultaneously to test database locking

## Positive Observations

The code demonstrates many **excellent practices**:

✅ **Security:** Proper SQL parameterization prevents injection attacks  
✅ **Resource Management:** Consistent use of context managers for files  
✅ **Type Safety:** Comprehensive type hints throughout  
✅ **Error Handling:** Specific exception types, no bare except clauses  
✅ **Documentation:** Good docstrings and inline comments  
✅ **Data Validation:** Extensive range checking in config  
✅ **Separation of Concerns:** Well-organized module structure  
✅ **Testing:** Comprehensive test suite (~1280 tests mentioned)

## Conclusion

This is a **well-engineered codebase** with only minor issues. The most critical concern is the database concurrency pattern with WAL mode. The unsafe array access patterns are worth fixing to prevent rare edge-case crashes, but the code generally has good defensive programming practices.

**Overall Risk Assessment:** LOW-MEDIUM

The codebase is production-ready with the caveat that database access should be single-threaded or properly synchronized.
