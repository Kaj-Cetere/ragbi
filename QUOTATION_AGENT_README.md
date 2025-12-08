# Pydantic-AI Quotation Agent

## Overview

The quotation agent uses **pydantic-ai** to intelligently quote specific seifim from Shulchan Aruch and commentaries (Ba'er Hetev, Mishna Berurah) within LLM responses. Instead of just citing sources in text, the agent can call tools to insert structured quotation blocks.

## How It Works

### Architecture

1. **Agent Definition** (`quotation_agent.py`):
   - Uses `pydantic-ai.Agent` with structured output type `QuotationResponse`
   - Provides two tools: `quote_seif` and `quote_commentary`
   - Configured with OpenRouter's Gemini model

2. **Tools**:
   - **`quote_seif(sefaria_ref)`**: Quotes a specific seif from Shulchan Aruch
   - **`quote_commentary(base_ref, commentator_name)`**: Quotes commentary on a seif

3. **Structured Output**:
   ```python
   class QuotationResponse(BaseModel):
       answer: str  # Main answer text
       seif_quotations: list[SeifQuotation]  # Quoted seifim
       commentary_quotations: list[CommentaryQuotation]  # Quoted commentaries
   ```

### Tool Behavior

When the LLM wants to quote a source, it calls the appropriate tool:

```python
# Example: Agent decides to quote a seif
quote_seif(sefaria_ref="Shulchan Arukh, Orach Chayim 5:1")

# Example: Agent decides to quote commentary
quote_commentary(
    base_ref="Shulchan Arukh, Orach Chayim 5:1",
    commentator_name="Mishnah Berurah"
)
```

The tools:
1. First check the retrieved RAG chunks for the source
2. If not found, query the Supabase database
3. If still not found, fetch from Sefaria API (for commentaries)
4. Return structured quotation data

## Usage

### In Streamlit App

1. Enable the quotation agent in the sidebar:
   - Toggle "📜 Enable Quotation Agent"

2. Ask a question as normal

3. The agent will:
   - Analyze the retrieved sources
   - Generate an answer
   - Call tools to quote relevant sources
   - Return formatted response with quotation blocks

### Programmatic Usage

```python
from quotation_agent import generate_response_with_quotations_sync
from supabase import create_client

# Setup
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Generate response
response = generate_response_with_quotations_sync(
    query="What does Shulchan Arukh say about saying Hashem's name?",
    context=rag_context,  # From your RAG retrieval
    retrieved_chunks=chunks,  # Raw chunks from vector search
    supabase=supabase,
    openrouter_api_key=OPENROUTER_API_KEY
)

# Access structured output
print(response.answer)
print(f"Quoted {len(response.seif_quotations)} seifim")
print(f"Quoted {len(response.commentary_quotations)} commentaries")

# Format for display
formatted = format_quotation_response(response)
print(formatted)
```

## Benefits

1. **Structured Quotations**: Quotations are separate from narrative text
2. **Verifiable Sources**: Each quotation includes full reference metadata
3. **Intelligent Selection**: LLM decides which sources to quote based on relevance
4. **Consistent Formatting**: All quotations formatted uniformly
5. **Extensible**: Easy to add more quotation types (e.g., Rambam, Tur)

## Example Output

```markdown
According to the Shulchan Arukh, one should be careful when pronouncing 
Hashem's name. The Mishna Berurah provides important clarification on this matter.

---

### 📜 Quoted Sources

**Shulchan Arukh, Orach Chayim 5:1**

> צריך ליזהר מאד בהזכרת השם שלא יזכירנו לבטלה...

---

### 💬 Commentaries

**Mishnah Berurah** on Shulchan Arukh, Orach Chayim 5:1

> ליזהר מאד - דהוי איסור חמור מאד...
```

## Testing

Run the test script:

```bash
python test_quotation_agent.py
```

This will:
1. Fetch sample chunks from the database
2. Run the agent with a test query
3. Display structured output and formatted response

## Technical Details

### Pydantic Models

- **`SeifQuotation`**: Contains ref, text, book, siman, seif
- **`CommentaryQuotation`**: Contains ref, commentator, text, commentary_ref
- **`QuotationResponse`**: Main output with answer and quotation lists
- **`AgentDeps`**: Runtime dependencies (Supabase client, chunks)

### Configuration

The agent uses:
- Model: `openrouter/google/gemini-2.5-flash-lite-preview-09-2025`
- System prompt: Instructs agent on when/how to use quotation tools
- Output type: Structured `QuotationResponse`

### Error Handling

- Falls back to standard streaming if agent fails
- Graceful degradation if sources not found
- Placeholder text for missing quotations

## Future Enhancements

Potential additions:
1. More quotation types (Rambam, Tur, Rema)
2. Cross-reference detection
3. Quotation validation
4. Citation graph generation
5. Multi-language support (English translations)
