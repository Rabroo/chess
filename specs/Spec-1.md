**Universal Data Scraper – Technical Specification**

## **Overview**

This project defines a modular, extensible scraping system capable of collecting diverse data types: images, chess positions, numeric relationships, scientific/experimental datasets, and social media content. The system is designed to support additional data categories in the future with minimal structural changes. Scraped output is stored in a new directory structure created by the tool on first run. Duplicate detection is required across all data types.

---

## **Directory Structure**

On first run, the tool must create the following root directory if absent:

raw/

Under this directory, each scraper module stores output in its own subfolder:

raw/  
    images/  
    chess\_positions/  
    numeric\_relations/  
    experiments/  
    social/  
    meta/

The structure must be auto-generated based on available scraper modules. New modules added later will automatically create a new subfolder of the same name.

---

## **Command-Line Interface**

### **Base Command**

scrape \--type \<module\> \--input \<value\> \[options\]

### **Arguments**

| Argument | Required | Description |
| ----- | ----- | ----- |
| `--type` | yes | Specifies the scraper module (e.g., images, chess, numeric, experiments, social) |
| `--input` | yes | Module-specific input (URL, query, term, FEN string, dataset reference, etc.) |
| `--limit` | no | Max number of items to scrape (default depends on type) |
| `--output-dir` | no | Override default directory within `raw/` |
| `--no-duplicates` | no | Enforce duplicate filtering (on by default) |
| `--log` | no | Path to log file (default: `raw/meta/scraper.log`) |

---

## **Module Definitions**

### **1\. Image Scraper**

**Description:** Downloads images from URLs, queries, or direct links.

**Input:**

* URL  
* search query  
* list of image URLs

**Output Format:**

* Saved as `.jpg` or `.png` in `raw/images/`

**Duplicate Detection:**

* SHA256 hash comparison  
* Filename normalization

---

### **2\. Chess Position Scraper**

**Description:** Collects chess positions from online databases or input FEN strings.

**Input:**

* FEN string  
* URL to a PGN or chess database

**Output Format:**

* `.fen` files stored in `raw/chess_positions/`  
* JSON metadata per position

**Duplicate Detection:**

* Exact FEN string match

---

### **3\. Numeric Relation Scraper**

**Description:** Collects structured numerical relationships (e.g., sequences, function relationships, OEIS-style data).

**Input:**

* sequence name  
* sequence ID  
* formula/relationship query

**Output Format:**

* JSON entries stored in `raw/numeric_relations/`

**Duplicate Detection:**

* Hash of normalized representation

---

### **4\. Experiment Dataset Scraper**

**Description:** Fetches structured experimental datasets or reports from public sources.

**Input:**

* dataset URL  
* experiment keyword  
* dataset ID

**Output Format:**

* JSON, CSV, or downloaded files stored in `raw/experiments/`

**Duplicate Detection:**

* File checksums  
* Metadata signature hash

---

### **5\. Social Media Scraper**

**Description:** Collects posts, metadata, and media from supported public APIs.

**Input:**

* search term  
* hashtag  
* user handle  
* post URL

**Output Format:**

* JSON per post in `raw/social/`  
* Images/media stored alongside

**Duplicate Detection:**

* Post ID or canonical URL

---

## **Logging**

All modules must log to:

raw/meta/scraper.log

Log format:

\[TIMESTAMP\] MODULE:ACTION:DETAILS

Examples:

* creation of directories  
* start/end of scraping job  
* number of items collected  
* duplicate items skipped  
* errors encountered

---

## **Duplicate Handling**

Duplicates must be prevented using:

1. **Content hashing** (SHA256)  
2. **Canonical path normalization**  
3. **Metadata comparison** (when applicable)

Duplicate items must not be downloaded again. Instead, a log entry must be created:

\[DATE\] SKIPPED\_DUPLICATE: \<identifier\>

---

## **Extensibility Requirements**

The scraper must support future additions with zero architectural changes. Requirements:

* Each scraper is a self-contained module in `scrapers/`  
* Modules auto-register through a loader script  
* Adding a new module automatically:  
  * Registers a new `--type` option  
  * Creates a new folder under `raw/`  
  * Enables custom input format validation  
  * Enables custom scraping logic

---

## **Recommended Code Structure**

scraper/  
    main.py  
    utils/  
        hashing.py  
        downloader.py  
        logger.py  
        directory.py  
    scrapers/  
        images.py  
        chess\_positions.py  
        numeric\_relations.py  
        experiments.py  
        social.py

Each module implements a common interface:

class ScraperModule:  
    def validate\_input(self, input):  
        pass

    def fetch(self, input, limit):  
        pass

    def store(self, item):  
        pass

---

## **Scraper Workflow**

1. Parse CLI arguments  
2. Load module based on `--type`  
3. Validate input  
4. Fetch data items  
5. For each item:  
   * generate content hash  
   * check for duplicate  
   * store item  
   * log success  
6. Final summary printed to console

---

## **Output Summary**

After completion, the script must print:

Scraping complete.  
Items processed: X  
Duplicates skipped: Y  
Items saved: Z  
Output directory: raw/\<module\>  
