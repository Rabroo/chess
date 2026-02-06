# **Universal Data Scraper – Technical Specification (Revised)**

## **Overview**

This project defines a modular, extensible scraping system capable of gathering a wide range of data types including images, chess positions, numeric relationships, experimental datasets, social media content, and future extensions. The system initializes its own directory structure, prevents duplication, validates inputs, handles errors and rate limits, supports authentication when applicable, and uses a centralized configuration system. All modules follow a common interface and must support retry logic, validation, and asynchronous or parallel execution when appropriate.

---

# **1\. Directory Structure**

On first run, the scraper must create:

raw/  
    images/  
    chess\_positions/  
    numeric\_relations/  
    experiments/  
    social/  
    meta/  
config/  
logs/

Future modules automatically create new subdirectories under `raw/`.

---

# **2\. Configuration System**

The scraper must support:

* `config/default.yaml`  
* `config/<module>.yaml`  
* Environment variable overrides  
* CLI overrides

### **Example configuration fields:**

network:  
  timeout: 10  
  retries: 3  
  retry\_backoff: 2  
  user\_agent: "UniversalScraper/1.0"

rate\_limits:  
  global\_max\_rps: 2  
  images\_max\_rps: 1  
  social\_max\_rps: 0.5

storage:  
  max\_file\_size\_mb: 50  
  disk\_space\_min\_mb: 500  
  log\_rotation\_mb: 10  
  log\_backup\_count: 5

---

# **3\. Command-Line Interface**

scrape \--type \<module\> \--input \<value\> \[options\]

### **Arguments**

| Argument | Required | Description |
| ----- | ----- | ----- |
| `--type` | yes | Module name (images, chess, numeric, experiments, social) |
| `--input` | yes | Module-specific input value |
| `--limit` | no | Max number of items to scrape (default from config) |
| `--config` | no | Path to custom config file |
| `--output-dir` | no | Overrides default directory |
| `--async` | no | Enables concurrent fetching |
| `--no-duplicates` | no | Enforces duplicate filter (default on) |
| `--log` | no | Custom log path |

---

# **4\. Error Handling Requirements**

All modules must implement:

### **Network Timeouts**

* Configurable via `network.timeout`  
* Hard fail after timeout

### **Retry Logic**

* Configurable retry attempts  
* Exponential backoff (`retry_backoff`)

### **Rate Limiting**

* Global and per-module limits  
* Auto-adjusting if API returns 429

### **Graceful Degradation**

If a dataset partially fails:

* log the error  
* skip the failing item  
* continue execution

---

# **5\. Authentication & Credential Management**

For modules requiring credentials (e.g., social media):

* Credentials are never hardcoded  
* Credentials are loaded from:  
  * `config/credentials.yaml`  
  * Environment variables  
  * OS keychain (optional extension)

Format:

social:  
  twitter\_api\_key: ${TWITTER\_API\_KEY}  
  twitter\_api\_secret: ${TWITTER\_API\_SECRET}

The scraper must detect missing credentials and fail with a clear error.

---

# **6\. Concurrency & Safety**

If `--async` is enabled, modules must support:

* Task queue with max concurrency from config  
* Thread-safe hashing for duplicate detection  
* Thread-safe write operations  
* Controlled rate limiting per worker

All filesystem writes must use locks when concurrency is active.

---

# **7\. Data Validation Requirements**

Each module must implement:

### **Input Validation**

* URLs validated to prevent SSRF  
* FEN strings validated for chess  
* JSON schema validation for experiments/numeric data  
* Character filtering on search terms

### **Content Validation**

* Content-type verification  
* Minimum data structure requirements  
* Hash integrity checks

Malformed or partial data must be:

* skipped  
* logged  
* not stored

---

# **8\. Storage Rules**

### **Disk Space Check**

Before writing any data, available disk space must be verified:

* Abort if less than `storage.disk_space_min_mb`

### **File Size Limits**

Reject files exceeding `max_file_size_mb`.

### **Log Rotation**

When log exceeds size threshold:

* rotate with timestamp  
* keep only `log_backup_count` previous logs

### **Large Dataset Handling**

If dataset exceeds memory threshold:

* stream downloads  
* write incrementally to disk

---

# **9\. Module-Specific Requirements**

## **Images Module**

* Validates URLs  
* Supports content-type: image/jpeg, image/png  
* Rejects files above size limit  
* Duplicate detection using:  
  * SHA256 content hash  
  * Normalized filename

---

## **Chess Positions Module**

* Validates FEN strings  
* Ensures legal piece placement  
* Outputs `.fen` plus metadata JSON  
* Duplicate detection via canonical FEN

---

## **Numeric Relations Module**

* Validates numeric sequences  
* Ensures monotonic or defined pattern  
* Schema enforced via JSON schema  
* Duplicate detection via normalized representation hash

---

## **Experiments Module**

* Validates dataset URLs  
* Confirms allowed content-type  
* Ensures metadata completeness  
* Duplicate detection via metadata signature

---

## **Social Module**

* Requires OAuth or API keys  
* Validates URLs to prevent SSRF  
* Enforces rate limits strictly  
* Duplicate detection via post ID

---

# **10\. Security Requirements**

All modules must meet the following:

### **URL Validation**

* Block internal IP ranges (SSRF protection)  
* Only HTTP/S allowed  
* Reject file:// and custom schemes

### **Content-Type Verification**

* Inspect headers before download  
* Inspect file magic bytes after download

### **Sandboxing**

* Files downloaded to a temp directory first  
* Validated before moved into `raw/`

### **Sanitization**

* Remove unsafe characters from filenames

---

# **11\. Duplicate Handling**

Duplicate detection uses:

1. **SHA256 hashing**  
2. **Canonical path**  
3. **Metadata signature (if applicable)**

If duplicate:

* skip storing  
* log duplicate event  
* do not download again if URL repeats

---

# **12\. Testing Requirements**

The project must include:

### **Unit Tests**

* Input validation  
* Hashing  
* Directory creation  
* Config parsing

### **Integration Tests**

* Real network requests (optional)  
* Mocked network responses (default)  
* OAuth flow mocks for social media

### **Stress Tests**

* High volume scraping  
* Concurrency under load

### **Security Tests**

* SSRF validation  
* Malformed files  
* Unauthorized credentials

Mocking guidelines:

* Network requests must use stubs  
* Filesystem writes must use temp folders

---

# **13\. Dependency Specification**

### **Language**

* Python 3.10+

### **Required Libraries**

requests  
aiohttp  
pydantic  
pyyaml  
tqdm  
python-dateutil  
hashlib (built-in)  
pathlib (built-in)  
logging (built-in)  
pytest  
aiofiles

### **Optional**

authlib (for OAuth)  
beautifulsoup4 (for HTML parsing)  
opencv-python (for image validation)

---

# **14\. Code Structure**

scraper/  
    main.py  
    config/  
        default.yaml  
        credentials.yaml  
    logs/  
    utils/  
        hashing.py  
        downloader.py  
        validator.py  
        logger.py  
        directory.py  
        rate\_limiter.py  
        async\_tools.py  
    scrapers/  
        base.py  
        images.py  
        chess\_positions.py  
        numeric\_relations.py  
        experiments.py  
        social.py  
tests/  
    unit/  
    integration/  
    mocks/

---

# **15\. Workflow Summary**

1. Load config  
2. Merge environment variable overrides  
3. Parse CLI arguments  
4. Validate module type  
5. Validate input  
6. Authenticate if required  
7. Initialize rate limiter  
8. Create directories  
9. Begin scraping loop  
10. Handle retries, errors, timeouts  
11. Perform duplicate detection  
12. Validate content-type  
13. Store files safely  
14. Log results  
15. Print summary