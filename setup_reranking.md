# Quick Setup Guide for Reranking

## Step 1: Install Dependencies

```bash
pip install zeroentropy
```

Or reinstall all dependencies:
```bash
pip install -r requirements.txt
```

## Step 2: Get Your ZeroEntropy API Key

1. Visit https://dashboard.zeroentropy.dev
2. Create an account or sign in
3. Generate a new API key

## Step 3: Add API Key to Environment

Add to your `.env` file:
```bash
ZEROENTROPY_API_KEY=your_api_key_here
```

## Step 4: Test the Integration

Run the test script to verify reranking works:
```bash
python test_retrieval.py
```

Expected output should show comparison between with/without reranking.

## Step 5: Run the App

```bash
streamlit run app.py
```

## Step 6: Configure Reranking

In the Streamlit sidebar, you'll see a new "Reranking" section:

- **Enable Reranking**: Toggle to turn on/off
- **Initial Retrieval Count**: Set to 60 (retrieves 60 chunks before reranking)
- **Top K After Rerank**: Set to 10 (keeps top 10 after reranking)
- **Chunks to Hydrate**: Set to 5 (hydrates top 5 with commentaries)

## Verification Checklist

- [ ] `zeroentropy` package installed
- [ ] `ZEROENTROPY_API_KEY` added to `.env`
- [ ] `test_retrieval.py` runs without errors
- [ ] Streamlit app shows reranking toggle in sidebar
- [ ] Query logs show reranking time in performance summary
- [ ] Source chunks display rerank scores when enabled

## Troubleshooting

### "ZEROENTROPY_API_KEY not found"
- Check your `.env` file has the key
- Restart your terminal/IDE to reload environment variables

### "Failed to initialize ZeroEntropy client"
- Verify API key is valid
- Check internet connection
- Try creating a new API key

### Reranking seems slow
- Normal: Reranking typically adds 0.5-2 seconds per query
- Reduce initial retrieval count if needed (60 → 40)

### No improvement in results
- Increase initial retrieval count (60 → 80)
- Check that relevant documents exist in your database
- Review test cases to ensure expected refs are correct

## Next Steps

1. **Test with your queries**: Try various questions to see reranking impact
2. **Tune parameters**: Adjust retrieval counts based on your needs
3. **Monitor costs**: Track API usage on ZeroEntropy dashboard
4. **Compare modes**: Use the comparison mode to see regular vs contextual embeddings with reranking

## Support

- Documentation: See `RERANKING_GUIDE.md` for detailed information
- ZeroEntropy Docs: https://docs.zeroentropy.dev
- Issues: Contact founders@zeroentropy.dev
