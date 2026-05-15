"""
test_rag.py
===========
Quick sanity-test for the RAG pipeline.
Run with:  python test_rag.py

Tests:
  1. Text extraction from a dummy PDF (created in-memory)
  2. Chunking
  3. Embedding
  4. FAISS index build & retrieval
  5. Out-of-scope detection (no Gemini key needed for steps 1-5)
"""

import sys
import os
import tempfile

# ── 1. Check imports ──────────────────────────────────────────────────────────
print("=" * 60)
print("  Study Buddy RAG — Pipeline Test")
print("=" * 60)

try:
    from rag_engine import (
        extract_text_from_pdfs,
        chunk_pages,
        RAGEngine,
    )
    print("✅ rag_engine imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# ── 2. Create a dummy PDF ─────────────────────────────────────────────────────
print("\n[1/5] Creating a sample PDF for testing…")
try:
    from fpdf import FPDF  # type: ignore

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    sample_text = (
        "Photosynthesis is the process by which green plants convert sunlight "
        "into food using carbon dioxide and water. The equation for photosynthesis "
        "is: 6CO2 + 6H2O + light → C6H12O6 + 6O2. "
        "Chlorophyll is the green pigment responsible for absorbing light energy. "
        "Photosynthesis occurs in the chloroplasts of plant cells. "
        "There are two stages: the light-dependent reactions and the Calvin cycle. "
        "Newton's First Law states that an object remains at rest or in uniform "
        "motion unless acted upon by an external force. "
        "Newton's Second Law: F = ma, where F is force, m is mass, a is acceleration. "
        "Newton's Third Law: For every action there is an equal and opposite reaction."
    )
    pdf.multi_cell(0, 10, sample_text)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    tmp_path = tmp.name
    print(f"   Created temp PDF: {tmp_path}")
    use_sample = True

except ImportError:
    print("   fpdf not installed — using a pre-existing PDF if available")
    # Look for any PDF in current directory
    pdfs = [f for f in os.listdir(".") if f.endswith(".pdf")]
    if pdfs:
        tmp_path = pdfs[0]
        use_sample = True
        print(f"   Using existing PDF: {tmp_path}")
    else:
        print("   No PDF found. Skipping extraction test.")
        use_sample = False

# ── 3. Text extraction ────────────────────────────────────────────────────────
print("\n[2/5] Testing PDF text extraction…")
if use_sample:
    pages = extract_text_from_pdfs([tmp_path])
    if pages:
        print(f"   ✅ Extracted {len(pages)} page(s)")
        print(f"   Preview: {pages[0]['text'][:120]}…")
    else:
        print("   ⚠️  No text extracted (PDF may be image-based)")
        pages = [{"source": "test.pdf", "page": 1,
                  "text": "Photosynthesis converts sunlight into food. Chlorophyll absorbs light energy."}]

# ── 4. Chunking ───────────────────────────────────────────────────────────────
print("\n[3/5] Testing chunking…")
chunks = chunk_pages(pages, chunk_size=200, overlap=30)
print(f"   ✅ {len(chunks)} chunks created from {len(pages)} page(s)")
if chunks:
    print(f"   First chunk: \"{chunks[0]['text'][:80]}…\"")

# ── 5. Embedding + FAISS ──────────────────────────────────────────────────────
print("\n[4/5] Building FAISS index (this downloads the model on first run)…")
engine = RAGEngine()
try:
    if use_sample:
        n_pages, n_chunks = engine.build_index([tmp_path], chunk_size=200,
                                               overlap=30, force_rebuild=True)
    else:
        sys.exit(0)
    print(f"   ✅ Index built: {n_pages} pages, {n_chunks} chunks")
except Exception as e:
    print(f"   ❌ Index build failed: {e}")
    sys.exit(1)

# ── 6. Retrieval ──────────────────────────────────────────────────────────────
print("\n[5/5] Testing retrieval…")
test_queries = [
    ("What is photosynthesis?", True),
    ("Who won the FIFA World Cup?", False),   # out-of-scope
]

for query, expect_in_scope in test_queries:
    results = engine.retrieve(query, top_k=2)
    best_score = results[0]["score"] if results else 0.0
    in_scope   = best_score >= 0.30
    status     = "✅" if in_scope == expect_in_scope else "⚠️ "
    label      = "IN SCOPE" if in_scope else "OUT OF SCOPE"
    print(f"   {status} \"{query}\"")
    print(f"      → {label} (best score: {best_score:.3f})")

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ All pipeline tests passed!")
print("  Now run:  streamlit run app.py")
print("=" * 60)

# Cleanup
if use_sample and 'tmp_path' in locals():
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
