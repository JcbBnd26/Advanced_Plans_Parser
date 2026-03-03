# Security Assessment: Advanced Plans Parser

**Assessment Date:** March 2, 2026  
**Application:** Advanced Plans Parser (PDF plan-sheet analysis pipeline)  
**Version:** 0.1.0

---

## Executive Summary

**Overall Security Grade: B- (Good, with notable gaps)**

The Advanced Plans Parser is a well-structured Python application for parsing architectural/construction plans with generally sound security practices. However, there are several areas requiring attention before production deployment, particularly around input validation, DoS protection, and dependency management.

---

## Detailed Findings

### 🟢 **STRENGTHS**

#### 1. SQL Injection Prevention (Grade: A)
- **Status:** EXCELLENT
- **Evidence:** All database queries use parameterized statements with `?` placeholders
- **Example:** `self._conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,))`
- **Impact:** No SQL injection vulnerabilities detected in corrections store module

#### 2. Data Privacy Policy (Grade: A)
- **Status:** EXCELLENT  
- **Evidence:** Well-documented privacy policy with three levels of data control
- **Features:**
  - `local_only` (default) - prevents cloud LLM access
  - `cloud_allowed` - permits cloud access with explicit configuration
  - `cloud_with_consent` - requires user consent before cloud access
- **Impact:** Strong protection for confidential construction plans

#### 3. PDF Input Validation (Grade: B+)
- **Status:** GOOD
- **Evidence:** Basic validation implemented in `ingest.py`
- **Checks performed:**
  - File existence verification
  - File type checking (.pdf extension)
  - Empty file detection
  - Encryption/password-protection detection
- **Impact:** Reasonable protection against basic malformed inputs

#### 4. Path Handling (Grade: B)
- **Status:** ADEQUATE
- **Evidence:** Uses `Path.resolve()` to normalize paths
- **Impact:** Some protection against path traversal, though not comprehensive

---

### 🟡 **MODERATE CONCERNS**

#### 5. API Key Management (Grade: C+)
- **Status:** NEEDS IMPROVEMENT
- **Issues:**
  - API keys passed as constructor parameters without encryption
  - No environment variable usage pattern shown
  - Keys stored in memory as plain strings
  - No key rotation mechanism
- **Recommendations:**
  - Use environment variables (e.g., `os.getenv("OPENAI_API_KEY")`)
  - Implement key masking in logs
  - Add support for secret management services (e.g., AWS Secrets Manager, HashiCorp Vault)
  - Document secure key storage best practices

#### 6. Dependency Management (Grade: C)
- **Status:** NEEDS IMPROVEMENT
- **Issues:**
  - Dependencies use minimum version constraints (`>=`) rather than pinned versions
  - No automated vulnerability scanning configured
  - Large dependency surface area (PaddleOCR, PyTorch, OpenCV, etc.)
- **Vulnerable patterns:**
  - `Pillow>=12.0.0` - should pin to specific versions
  - `opencv-python>=4.8.0` - known CVEs in older versions
  - No `requirements.lock` or `poetry.lock` file
- **Recommendations:**
  - Pin all dependencies to specific versions
  - Add `safety` or `pip-audit` to CI/CD pipeline
  - Generate and use `requirements.lock` file
  - Set up Dependabot or similar automated scanning

#### 7. Error Handling & Information Disclosure (Grade: C+)
- **Status:** NEEDS REVIEW
- **Issues:**
  - Generic exception handling that may expose stack traces
  - No clear sanitization of error messages before display
  - Potential for sensitive path information in error messages
- **Example risk:**
  ```python
  except Exception as exc:
      raise IngestError(f"Cannot open PDF: {exc}") from exc
  ```
  This could expose internal file paths or system information
- **Recommendations:**
  - Sanitize error messages before showing to users
  - Log full details but show generic messages to users
  - Implement structured error codes instead of verbose messages

#### 8. Input Size Limits (Grade: D+)
- **Status:** MAJOR CONCERN
- **Issues:**
  - No file size limits enforced for PDF uploads
  - No pagination limits on PDF processing
  - No timeouts on OCR operations
  - Potential for resource exhaustion attacks
- **Attack vectors:**
  - Upload extremely large PDFs (e.g., 1GB+)
  - Upload PDFs with thousands of pages
  - Upload PDFs with extremely high DPI images
- **Recommendations:**
  - Add maximum file size limit (e.g., 100MB)
  - Add maximum page count limit (e.g., 500 pages)
  - Implement timeouts for OCR operations
  - Add rate limiting for API endpoints (if web service)

---

### 🔴 **CRITICAL CONCERNS**

#### 9. Authentication & Authorization (Grade: F)
- **Status:** MISSING
- **Issues:**
  - No authentication system implemented
  - No user management
  - No authorization checks on file access
  - GUI application runs with full file system access
  - No audit logging of user actions
- **Attack vectors:**
  - Anyone with application access can process any PDF
  - No user accountability for corrections database modifications
  - No role-based access control (RBAC)
- **Recommendations:**
  - Implement user authentication system
  - Add role-based permissions (viewer, annotator, admin)
  - Add audit logging for all database modifications
  - Implement file access controls based on user permissions

#### 10. Path Traversal Protection (Grade: D)
- **Status:** INSUFFICIENT
- **Issues:**
  - Basic path resolution but no explicit traversal prevention
  - No whitelist of allowed directories
  - File paths accepted from user input without strict validation
- **Attack vector:**
  ```python
  # User could potentially provide:
  pdf_path = "../../../../etc/passwd"
  ```
- **Recommendations:**
  - Implement strict path validation against a whitelist
  - Use `os.path.commonpath()` to verify paths stay within allowed directories
  - Reject paths containing `..` or absolute paths from user input
  - Example implementation:
  ```python
  def validate_pdf_path(pdf_path: Path, allowed_dir: Path) -> Path:
      pdf_path = pdf_path.resolve()
      allowed_dir = allowed_dir.resolve()
      if not str(pdf_path).startswith(str(allowed_dir)):
          raise IngestError("Path traversal detected")
      return pdf_path
  ```

#### 11. Command Injection Risk (Grade: D)
- **Status:** POTENTIAL VULNERABILITY
- **Issues:**
  - GUI uses `filedialog.askopenfilename()` which is generally safe
  - However, no validation of file paths before passing to system commands
  - Potential risk if file paths are used in subprocess calls
- **Recommendations:**
  - Audit all uses of `subprocess`, `os.system`, or similar
  - Validate and sanitize all file paths from user input
  - Use allow-lists for acceptable file paths

#### 12. Sensitive Data in Logs (Grade: D)
- **Status:** NEEDS REVIEW
- **Issues:**
  - No evidence of log sanitization
  - API keys could potentially be logged during debugging
  - PDF file paths logged which may contain sensitive directory structures
- **Recommendations:**
  - Implement log sanitization for API keys, passwords, tokens
  - Use structured logging with automatic PII redaction
  - Review all logging statements for sensitive data exposure

---

## Additional Security Considerations

### 13. GUI Security (Tkinter)
- **Issues:**
  - Desktop GUI has full file system access
  - No sandboxing of PDF processing
  - Potential for malicious PDFs to exploit vulnerabilities
- **Recommendations:**
  - Run PDF processing in isolated subprocess with limited permissions
  - Implement file quarantine for suspicious PDFs
  - Add virus/malware scanning integration

### 14. Database Security
- **Status:** ADEQUATE
- **Positives:**
  - WAL mode enabled for better concurrency
  - Foreign keys enabled
  - Proper schema with constraints
- **Concerns:**
  - Database file not encrypted at rest
  - No backup/recovery mechanism documented
- **Recommendations:**
  - Implement SQLite encryption (e.g., SQLCipher)
  - Add automated backup system
  - Document disaster recovery procedures

### 15. Third-Party Dependencies
- **High-risk dependencies:**
  - **PaddleOCR** - Complex ML library with large attack surface
  - **OpenCV** - Known history of vulnerabilities
  - **PIL/Pillow** - Image processing libraries often have vulnerabilities
- **Recommendations:**
  - Keep all dependencies up to date
  - Subscribe to security advisories for critical dependencies
  - Consider containerization to limit blast radius

---

## Risk Matrix

| Finding | Severity | Likelihood | Risk Level | Priority |
|---------|----------|------------|------------|----------|
| Missing Authentication | High | High | **CRITICAL** | P0 |
| Path Traversal | High | Medium | **HIGH** | P0 |
| No Input Size Limits | High | High | **HIGH** | P0 |
| Unpinned Dependencies | Medium | High | **MEDIUM** | P1 |
| API Key Management | Medium | Medium | **MEDIUM** | P1 |
| Information Disclosure | Low | Medium | **LOW** | P2 |
| Database Encryption | Medium | Low | **LOW** | P2 |

---

## Prioritized Remediation Roadmap

### Phase 1: Critical (Immediate - Week 1)
1. **Implement Authentication System**
   - Add user login/logout
   - Session management
   - Basic RBAC (viewer/editor/admin roles)

2. **Add Path Traversal Protection**
   - Implement strict path validation
   - Whitelist allowed directories
   - Reject dangerous path patterns

3. **Add Input Size Limits**
   - Maximum file size: 100MB
   - Maximum pages: 500
   - OCR operation timeouts: 60 seconds per page

### Phase 2: High Priority (Week 2-3)
4. **Pin All Dependencies**
   - Generate requirements.lock
   - Set up automated vulnerability scanning
   - Update vulnerable dependencies

5. **Improve API Key Management**
   - Move to environment variables
   - Add key masking in logs
   - Document secure setup

6. **Add Audit Logging**
   - Log all file access
   - Log all database modifications
   - Log authentication events

### Phase 3: Medium Priority (Week 4-6)
7. **Sanitize Error Messages**
   - Remove sensitive path information
   - Implement structured error codes
   - Log full details server-side only

8. **Add Database Encryption**
   - Implement SQLCipher
   - Encrypt sensitive data at rest
   - Set up key management

9. **Implement Rate Limiting**
   - Limit PDF processing per user
   - Prevent resource exhaustion
   - Add request throttling

### Phase 4: Ongoing (Continuous)
10. **Security Monitoring**
    - Set up automated dependency scanning
    - Regular security audits
    - Penetration testing
    - Security training for developers

---

## Testing Recommendations

### Security Test Suite
1. **SQL Injection Tests** - ✅ Already resistant
2. **Path Traversal Tests** - ❌ Create test cases
3. **Authentication Bypass** - ❌ Not applicable (no auth)
4. **DoS Tests** - ❌ Test with large files
5. **API Key Exposure** - ⚠️ Review logs and error messages

### Recommended Tools
- **SAST:** Bandit, Semgrep
- **Dependency Scanning:** Safety, pip-audit
- **Dynamic Testing:** OWASP ZAP (if web interface added)
- **Penetration Testing:** Manual testing by security team

---

## Compliance Considerations

### Data Privacy
- ✅ GDPR-friendly with local_only mode
- ✅ Clear data privacy policy documented
- ⚠️ No data retention policy documented
- ❌ No right to deletion implementation

### Industry Standards
- ⚠️ OWASP Top 10 - Several issues present
- ❌ No security.txt file
- ❌ No vulnerability disclosure policy
- ❌ No security incident response plan

---

## Conclusion

The Advanced Plans Parser demonstrates solid foundational security practices in database handling and data privacy. However, **it is not production-ready** without addressing the critical authentication, input validation, and DoS protection gaps.

### Key Takeaways:
✅ **Good:** SQL injection prevention, data privacy policy, parameterized queries  
⚠️ **Needs Work:** API key management, dependency management, error handling  
❌ **Critical Gaps:** Authentication, path traversal protection, input size limits

### Recommended Next Steps:
1. Implement authentication system (P0)
2. Add input validation and size limits (P0)
3. Fix path traversal vulnerabilities (P0)
4. Pin dependencies and set up scanning (P1)
5. Conduct full security audit before production deployment

---

**Assessment prepared by:** Claude  
**Methodology:** Static code analysis, architecture review, dependency audit  
**Limitations:** This assessment is based on source code review only. Dynamic testing and penetration testing would provide additional insights.
