# ESG MCP Servers

<!-- mcp-name: io.github.freminder/esg-mcp-servers -->

Open-source [Model Context Protocol](https://modelcontextprotocol.io/) servers for ESG (Environmental, Social, and Governance) data extraction, analysis, and regulation management.

**31 tools** across 6 servers — install once, run only what you need.

**Author:** Ioannis Michos ([johnmichos.tf@gmail.com](mailto:johnmichos.tf@gmail.com))

## Quick Start

```bash
pip install esg-mcp-servers
```

### Prerequisites

| Service | Required for |
|---------|-------------|
| PostgreSQL 16 + pgvector | Vector storage, metrics, regulations |
| MongoDB 7 | PDF binary storage (GridFS) |
| Anthropic API key | RAG queries, metric extraction |

Spin up local databases with Docker:

```bash
docker compose up -d
```

Run the database migration:

```bash
esg-mcp-migrate
```

### Environment Variables

Copy `.env.example` and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DSN` | `postgresql://esg:esg@localhost/esg_platform` | PostgreSQL connection string |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `ANTHROPIC_API_KEY` | — | Required for RAG and LLM extraction |
| `EMBEDDING_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | Sentence-transformer model |
| `EMBEDDING_DIMENSIONS` | `1024` | Embedding vector size |

## Servers & Tools

### esg-metrics-extractor (11 tools)

ESRS-aligned KPI extraction for emissions, energy, water, waste, social, and governance domains.

| Tool | Description |
|------|-------------|
| `extract_emissions_data` | GHG Scope 1, 2, 3 extraction |
| `extract_energy_data` | Energy consumption & renewables (ESRS E2) |
| `extract_water_data` | Water withdrawal, discharge, consumption (ESRS E3) |
| `extract_waste_data` | Waste generation, recycling, landfill (ESRS E5) |
| `extract_social_data` | Workforce, diversity, H&S (ESRS S1) |
| `extract_governance_data` | Board composition & governance (ESRS G1) |
| `answer_esg_query` | Free-text RAG Q&A over documents |
| `keyword_similarity_search` | Semantic keyword search in documents |
| `detect_query_domain` | Classify query into ESG domain |
| `detect_emissions_query_type` | Detect emissions scope/year from query |
| `batch_extract_metrics` | Extract all KPIs and persist to DB |

### esg-pdf-processor (5 tools)

PDF validation, text/table extraction, and embedding generation.

| Tool | Description |
|------|-------------|
| `verify_esg_report` | RandomForest ESG report classifier |
| `extract_text_chunks` | Extract and chunk PDF text |
| `extract_tables` | Table extraction with OCR fallback |
| `generate_embeddings` | Batch embedding generation |
| `process_pdf_full_pipeline` | End-to-end: extract, embed, store |

### esg-vector-store (5 tools)

pgvector CRUD operations for document chunks and query cache.

| Tool | Description |
|------|-------------|
| `upsert_document_chunks` | Insert/update chunks with embeddings |
| `similarity_search` | Cosine similarity search |
| `get_cached_query_response` | Retrieve cached LLM responses |
| `cache_query_response` | Store LLM response in cache |
| `list_documents` | List indexed documents |

### esg-regulations (5 tools)

EU ESG regulation download, ingestion, and semantic search.

| Tool | Description |
|------|-------------|
| `download_regulation` | Download regulation PDF from EUR-Lex |
| `download_all_regulations` | Batch download all configured regulations |
| `ingest_regulation` | Extract, parse articles, embed, store |
| `search_regulation_text` | Semantic search across regulation articles |
| `list_regulations` | List ingested regulations |

### esg-scraper (5 tools)

ESG report discovery and download from the web.

| Tool | Description |
|------|-------------|
| `search_esg_reports` | Multi-engine search for ESG PDFs |
| `crawl_company_website` | Deep-crawl website for PDF links |
| `download_pdf` | Download PDF and store in GridFS |
| `get_scraping_status` | Check scrape job status |
| `cancel_scraping_job` | Cancel running scrape job |

### esg-mcp-all (31 tools)

All tools from all servers in a single process.

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "esg-metrics": {
      "command": "esg-metrics-extractor"
    },
    "esg-pdf": {
      "command": "esg-pdf-processor"
    },
    "esg-vectors": {
      "command": "esg-vector-store"
    },
    "esg-regulations": {
      "command": "esg-regulations"
    },
    "esg-scraper": {
      "command": "esg-scraper"
    }
  }
}
```

Or use the combined server for all tools:

```json
{
  "mcpServers": {
    "esg": {
      "command": "esg-mcp-all"
    }
  }
}
```

## Optional: Scraper Dependencies

The scraper server requires Selenium for JavaScript-heavy sites:

```bash
pip install esg-mcp-servers[scraper]
```

## Development

```bash
git clone https://github.com/freminder/esg-mcp-servers.git
cd esg-mcp-servers
pip install -e ".[scraper,dev]"
docker compose up -d
esg-mcp-migrate
```

## License

MIT — see [LICENSE](LICENSE).
