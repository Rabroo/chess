# Universal Data Scraper - Project Plan

## Phase 1: Project Foundation

### 1.1 Project Setup
- [ ] Initialize Python project with `pyproject.toml`
- [ ] Set up virtual environment (Python 3.10+)
- [ ] Install core dependencies: requests, aiohttp, pydantic, pyyaml, tqdm, python-dateutil, pytest, aiofiles
- [ ] Install optional dependencies: authlib, beautifulsoup4
- [ ] Create base directory structure:
  ```
  scraper/
      main.py
      config/
      logs/
      utils/
      scrapers/
  tests/
      unit/
      integration/
      mocks/
  ```

### 1.2 Configuration System
- [ ] Create `config/default.yaml` with network, rate_limits, storage sections
- [ ] Create `config/credentials.yaml` template
- [ ] Implement config loader with YAML parsing
- [ ] Implement environment variable override logic
- [ ] Implement CLI argument override logic
- [ ] Add Pydantic models for config validation

---

## Phase 2: Core Utilities

### 2.1 Directory Management (`utils/directory.py`)
- [ ] Implement auto-creation of `raw/`, `config/`, `logs/` directories
- [ ] Implement dynamic subdirectory creation for modules
- [ ] Add disk space checking before writes
- [ ] Implement temp directory for sandboxed downloads

### 2.2 Logging (`utils/logger.py`)
- [ ] Implement structured logging with format `[TIMESTAMP] MODULE:ACTION:DETAILS`
- [ ] Add log rotation based on `storage.log_rotation_mb`
- [ ] Implement backup count limiting
- [ ] Create console and file handlers

### 2.3 Hashing (`utils/hashing.py`)
- [ ] Implement SHA256 content hashing
- [ ] Implement canonical path normalization
- [ ] Implement metadata signature hashing
- [ ] Add thread-safe hash storage for duplicate detection

### 2.4 Downloader (`utils/downloader.py`)
- [ ] Implement synchronous download with requests
- [ ] Implement async download with aiohttp
- [ ] Add configurable timeout handling
- [ ] Implement retry logic with exponential backoff
- [ ] Add streaming for large files
- [ ] Implement content-type header inspection

### 2.5 Validator (`utils/validator.py`)
- [ ] Implement URL validation with SSRF protection (block internal IPs)
- [ ] Block non-HTTP/S schemes (file://, etc.)
- [ ] Implement content-type verification (headers + magic bytes)
- [ ] Implement filename sanitization
- [ ] Add file size limit checking

### 2.6 Rate Limiter (`utils/rate_limiter.py`)
- [ ] Implement global rate limiting
- [ ] Implement per-module rate limiting
- [ ] Add 429 response detection and auto-adjustment
- [ ] Add thread-safe token bucket or leaky bucket algorithm

### 2.7 Async Tools (`utils/async_tools.py`)
- [ ] Implement task queue with configurable max concurrency
- [ ] Add thread-safe file write operations with locks
- [ ] Implement worker pool management

---

## Phase 3: Base Scraper Module

### 3.1 Base Class (`scrapers/base.py`)
- [ ] Define abstract `ScraperModule` class
- [ ] Define interface methods:
  - `validate_input(input) -> bool`
  - `fetch(input, limit) -> Iterator`
  - `store(item) -> bool`
  - `get_hash(item) -> str`
- [ ] Implement common duplicate detection logic
- [ ] Implement common error handling wrapper
- [ ] Add module auto-registration mechanism

---

## Phase 4: Scraper Modules

### 4.1 Images Module (`scrapers/images.py`)
- [ ] Implement URL validation
- [ ] Support content-types: image/jpeg, image/png
- [ ] Implement file size rejection
- [ ] Implement SHA256 + filename duplicate detection
- [ ] Store output to `raw/images/`

### 4.2 Chess Positions Module (`scrapers/chess_positions.py`)
- [ ] Implement FEN string validation
- [ ] Add legal piece placement verification
- [ ] Output `.fen` files + metadata JSON
- [ ] Implement canonical FEN duplicate detection
- [ ] Store output to `raw/chess_positions/`

### 4.3 Numeric Relations Module (`scrapers/numeric_relations.py`)
- [ ] Implement numeric sequence validation
- [ ] Add JSON schema enforcement
- [ ] Implement normalized representation hashing
- [ ] Store output to `raw/numeric_relations/`

### 4.4 Experiments Module (`scrapers/experiments.py`)
- [ ] Implement dataset URL validation
- [ ] Add content-type verification
- [ ] Implement metadata completeness checking
- [ ] Implement metadata signature duplicate detection
- [ ] Store output to `raw/experiments/`

### 4.5 Social Module (`scrapers/social.py`)
- [ ] Implement OAuth/API key authentication
- [ ] Add credential loading from config/env
- [ ] Implement SSRF-safe URL validation
- [ ] Enforce strict rate limiting
- [ ] Implement post ID duplicate detection
- [ ] Store output to `raw/social/`

---

## Phase 5: CLI & Main Entry Point

### 5.1 CLI Implementation (`main.py`)
- [ ] Implement argument parsing:
  - `--type` (required)
  - `--input` (required)
  - `--limit`
  - `--config`
  - `--output-dir`
  - `--async`
  - `--no-duplicates`
  - `--log`
- [ ] Implement module loader based on `--type`
- [ ] Wire up config loading sequence
- [ ] Implement main scraping workflow (steps 1-15 from spec)
- [ ] Add summary output on completion

---

## Phase 6: Testing

### 6.1 Unit Tests (`tests/unit/`)
- [ ] Test input validation for all modules
- [ ] Test hashing functions
- [ ] Test directory creation
- [ ] Test config parsing and overrides
- [ ] Test rate limiter logic
- [ ] Test URL/SSRF validation

### 6.2 Integration Tests (`tests/integration/`)
- [ ] Test full scraping workflow with mocked responses
- [ ] Test OAuth flow with mocks
- [ ] Test duplicate detection across runs
- [ ] Test error handling and retry logic
- [ ] Test async mode with concurrent workers

### 6.3 Mocks (`tests/mocks/`)
- [ ] Create network request stubs
- [ ] Create mock API responses per module
- [ ] Create temp filesystem fixtures

### 6.4 Security Tests
- [ ] Test SSRF protection with internal IP URLs
- [ ] Test malformed file handling
- [ ] Test missing/invalid credentials behavior

### 6.5 Stress Tests
- [ ] Test high-volume scraping (1000+ items)
- [ ] Test concurrency under load (10+ workers)

---

## Phase 7: Documentation & Finalization

### 7.1 Documentation
- [ ] Write README with usage examples
- [ ] Document configuration options
- [ ] Document how to add new scraper modules

### 7.2 Final Review
- [ ] Code review for security vulnerabilities
- [ ] Verify all spec requirements are met
- [ ] Performance profiling and optimization

---

## Dependencies Summary

### Required
```
requests
aiohttp
pydantic
pyyaml
tqdm
python-dateutil
pytest
aiofiles
```

### Optional
```
authlib
beautifulsoup4
opencv-python
```

---

## Implementation Order Recommendation

1. **Phase 1** - Foundation (config, structure)
2. **Phase 2** - Utilities (these are shared dependencies)
3. **Phase 3** - Base class (defines module contract)
4. **Phase 5** - CLI skeleton (enables testing modules)
5. **Phase 4** - Modules (start with Images as simplest, end with Social as most complex)
6. **Phase 6** - Testing (alongside module development)
7. **Phase 7** - Documentation and polish
