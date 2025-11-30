# 🏗️ Sefaria RAG: Technical Architecture & Implementation Guide

## 1. Project Overview
This project is a **Hybrid RAG (Retrieval-Augmented Generation)** search engine for Torah texts. It prioritizes accuracy and context by separating the "Base Text" (stored locally) from "Commentaries" (fetched live).

### Core Design Decisions
1.  **Storage:** Supabase (PostgreSQL) with `pgvector` for vector storage and metadata filtering.
2.  **Embeddings:** **Qwen 3 Embedding 8B** (`qwen/qwen3-embedding-8b`) via OpenRouter.
3.  **Client Library:** Official **OpenRouter Python SDK** (`openrouter`).
4.  **Dimensionality Strategy:** Uses **Matryoshka Representation Learning** via the API `dimensions` parameter to request 1536-dimensional vectors (down from 4096) to fit PostgreSQL page size limits.
5.  **Enrichment Strategy:** "Just-in-Time" Hydration. We do **not** embed commentaries. We fetch them via the Sefaria API only for the top retrieved chunks.

---

## 2. Data Pipeline & Ingestion

### A. Source Data
*   **Source:** [Sefaria-Export Repository](https://github.com/Sefaria/Sefaria-Export).
*   **Format:** `merged.json` files (Jagged Arrays).
*   **Excluded:** Do **not** use CLTK files.
*   **Folder Structure:**
    ```text
    /data
      /Shulchan_Arukh_Orach_Chayim/Hebrew/merged.json
      /Genesis/Hebrew/merged.json
    ```

### B. Ingestion Logic
The ingestion script (`ingest_supabase.py`) performs four critical transformations:

1.  **Cleaning:** Strips HTML tags (`<b>`, `<small>`) and Sefaria markers (`<i data-commentator>`).
2.  **Ref Generation:** Constructs the **Exact Sefaria Ref** needed for API lookups.
    *   *Format:* `"{Canonical Book Title} {Chapter}:{Segment}"`
    *   *Example:* `Shulchan Arukh, Choshen Mishpat 1:1`
3.  **Embedding:** Calls `client.embeddings.generate` using `qwen/qwen3-embedding-8b`.
4.  **Truncation:** Passes `dimensions=1536` to the API. The model natively shortens the vector to fit our database index.

### C. Database Schema (Supabase)
Run this SQL to configure the database:

```sql
-- Enable Vector Extension
create extension if not exists vector;

-- Main Table
create table sefaria_text_chunks (
  id uuid default gen_random_uuid() primary key,
  content text not null,                -- The cleaned Hebrew text
  embedding vector(1536),               -- Truncated Qwen vector
  sefaria_ref text not null,            -- Critical: Used for API Hydration
  metadata jsonb not null,              -- Filtering: { "book": "...", "siman": 1 }
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Indexing (IVFFlat)
create index on sefaria_text_chunks using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
```

---

## 3. Retrieval & Enrichment Workflow

The backend follows a strict 4-step process when answering a user query:

### Step 1: Vector Search
Query Supabase for the top $K$ segments (e.g., Top 10) based on the user's question.

### Step 2: Top-K Selection
Select only the **Top 3** results for "Hydration." The remaining results serve as background context without commentary.

### Step 3: API Hydration (The "Enricher")
For each of the Top 3 results, use the `sefaria_ref` column to call Sefaria:
*   **Endpoint:** `GET https://www.sefaria.org/api/texts/{sefaria_ref}?commentary=1&context=0`
*   **Context:** `0` ensures we get exactly that segment, not surrounding text.

### Step 4: Commentary Filtering
The API returns *all* commentaries. We must filter them programmatically using the **Allowed Commentators Map**.

#### 📜 Configuration: Allowed Commentators
*Business Logic: We only show specific classic commentaries for specific books.*

| Book Section | Allowed Commentaries (Exact `collectiveTitle` keys) |
| :--- | :--- |
| **Orach Chayim** | `Magen Avraham`, `Turei Zahav` (Taz) |
| **Yoreh De'ah** | `Siftei Kohen` (Shach), `Turei Zahav` (Taz) |
| **Choshen Mishpat** | `Siftei Kohen` (Shach), `Me'irat Einayim` (Sma) |
| **Even HaEzer** | `Chelkat Mechokek`, `Beit Shmuel` |
| **Talmud (Bavli)** | `Rashi`, `Tosafot` |

---

## 4. Implementation Details for Agents

### 🐍 Python: Ingestion Script Template
If modifying the ingestion logic, adhere to this structure using the SDK:

```python
from openrouter import OpenRouter

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
TARGET_DIMENSIONS = 1536

client = OpenRouter(api_key="...")

def get_embedding(text):
    # 1. Use the generate method
    res = client.embeddings.generate(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=TARGET_DIMENSIONS # Server-side truncation
    )
    return res.data[0].embedding
```

### 🐍 Python: Enrichment Script Template

```python
MAIN_COMMENTATORS = {
    "Shulchan Arukh, Choshen Mishpat": ["Siftei Kohen", "Me'irat Einayim"],
    # ... other books
}

def hydrate_chunk(chunk_ref, book_title):
    # 1. GET Sefaria API
    # 2. Parse ["commentary"] list
    # 3. Filter where c['collectiveTitle'] in MAIN_COMMENTATORS[book_title]
    # 4. Return enriched text block
    pass
```

## 5. Environment Variables
Ensure these are present in `.env`:

```bash
OPENROUTER_API_KEY="sk-or-..."      # For Qwen Embeddings
SUPABASE_URL="https://..."          # Database URL
SUPABASE_KEY="eyJ..."               # Service Role Key
```

## 6. Known Constraints & Edge Cases
1.  **Vector Dimensions:** Postgres throws error `54000` if dim > 2000. **Always use 1536.**
2.  **Sefaria Titles:** Sefaria is strict. `Shulchan Arukh` (with 'kh'), not `Aruch`. Use the `title` field from the JSON metadata to be safe.
3.  **Talmud Structure:** Talmud JSONs are often Daf-based. The standard parser works, but `ref` calculation must align with `Daf 2a` starting at index 0 or using the internal `ref` tags if available.
4.  **HTML Cleaning:** Do not remove the text *inside* `<small>` tags in Shulchan Aruch (that is the Rema). Only remove the tags themselves. Remove `<i data-commentator>` tags entirely.