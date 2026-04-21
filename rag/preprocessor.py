"""
preprocessor.py — Document preprocessing for RAG-optimised ingestion.

RAG retrieval quality depends heavily on how documents are formatted before
chunking.  Raw PDFs and Word docs typically contain:

  - Extraction artifacts (hyphenated line-breaks, ligatures, control chars)
  - Flat unstructured text with no chunk-boundary signals
  - No document-level context baked into individual chunks

This module transforms raw documents into a RAG-supporting format in four
stages:

  1. CLEAN         — remove PDF/OCR artifacts, normalise whitespace
  2. STRUCTURE     — detect section headers and inject normalised
                     === SECTION: <name> === markers that RecursiveChar-
                     acterTextSplitter uses as its top-priority split point
                     (configured via separators=["\n\n===", ...])
  3. ENRICH        — populate chunk metadata: title, doc_type,
                     section_count, word_count
  4. CONTEXTUALISE — after splitting, prefix every chunk with a compact
                     one-line context header:
                       [Document: X | KB: hr | Section: Annual Leave]
                     so every retrieved chunk is self-contained regardless
                     of where it appears in the source document

Result: instead of the LLM receiving an anonymous paragraph it always sees
the full provenance of each chunk — which document, KB, and section it
belongs to — enabling more accurate and grounded answers.
"""

import logging
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Control characters that survive PDF extraction but are not printable
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# PDF line-break hyphenation: "infor-\nmation" → "information"
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\n(\w)')

# Common Unicode ligatures not caught by NFKC normalisation
_LIGATURE_TABLE = str.maketrans({
    '\ufb01': 'fi',  # ﬁ
    '\ufb02': 'fl',  # ﬂ
    '\ufb00': 'ff',  # ﬀ
    '\ufb03': 'ffi', # ﬃ
    '\ufb04': 'ffl', # ﬄ
    '\ufb06': 'st',  # ﬆ
    '\u2019': "'",   # right single quote → apostrophe
    '\u201c': '"',   # left double quote  → straight quote
    '\u201d': '"',   # right double quote → straight quote
    '\u2013': '-',   # en-dash → hyphen
    '\u2014': '--',  # em-dash → double-hyphen
    '\u00a0': ' ',   # non-breaking space → space
})

# 3+ consecutive blank lines → 2 (one blank paragraph separator)
_MULTI_NL_RE = re.compile(r'\n{3,}')

# Trailing whitespace on each line
_TRAIL_RE = re.compile(r'[ \t]+$', re.MULTILINE)

# Repeated spaces (not newlines)
_MULTI_SP_RE = re.compile(r'[ \t]{2,}')

# Lone page-number lines (e.g. "- 12 -" or just "12" on its own line)
_PAGE_NUM_RE = re.compile(r'^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$', re.MULTILINE)


def clean_text(text: str) -> str:
    """Remove extraction artifacts and normalise whitespace.

    Handles ligatures, hyphenated line breaks, control characters, trailing
    whitespace, page-number lines, and redundant blank lines.  Safe to call
    on any document type; PDF output benefits most.
    """
    text = text.translate(_LIGATURE_TABLE)
    text = unicodedata.normalize('NFKC', text)
    text = _CTRL_RE.sub('', text)
    text = _HYPHEN_BREAK_RE.sub(r'\1\2', text)    # rejoin hyphenated words
    text = _PAGE_NUM_RE.sub('', text)              # strip bare page numbers
    text = _TRAIL_RE.sub('', text)
    text = _MULTI_SP_RE.sub(' ', text)
    text = _MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2a: Title extraction
# ─────────────────────────────────────────────────────────────────────────────

_MD_H1_RE   = re.compile(r'^#\s+(.+)',                  re.MULTILINE)
_UNDERLINE  = re.compile(r'^([^\n]{4,})\n[=]{3,}\s*$', re.MULTILINE)
_CAPS_TITLE = re.compile(r'^([A-Z][A-Z0-9 \-:]{4,60})\s*$', re.MULTILINE)
# First non-empty line that looks like a title: mixed-case, 4-80 chars, no
# sentence-ending punctuation (so we don't grab the first body sentence).
_FIRST_LINE_RE = re.compile(r'^([A-Z][^\n]{3,79}[^\.!?:,])\s*$', re.MULTILINE)


def extract_title(text: str, source: str, page: int | None = None) -> str:
    """Return a human-readable document title.

    Priority order:
      1. First Markdown H1 heading
      2. Underlined heading (Title\\n====)
      3. First non-empty mixed-case line in the opening 300 chars
      4. First ALL-CAPS line in the opening 500 chars (title-cased)
      5. Filename stem (underscores/hyphens → spaces, title-cased)

    When a page number is provided (PDF multi-page docs), the filename stem
    is used directly so each page doesn't pretend to have its own title.
    """
    if page is not None:
        stem = Path(source).stem if source else 'document'
        return re.sub(r'[_\-]+', ' ', stem).strip().title()

    m = _MD_H1_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _UNDERLINE.search(text)
    if m:
        return m.group(1).strip()

    # First line that looks like a document title (mixed-case, not a sentence)
    m = _FIRST_LINE_RE.search(text[:300])
    if m:
        candidate = m.group(1).strip()
        # Skip if it looks like a normal sentence (has a verb-like lowercase word)
        if not re.search(r'\b(is|are|was|were|the|a|an|to|of|and|or)\b', candidate):
            return candidate

    m = _CAPS_TITLE.search(text[:500])
    if m:
        return m.group(1).strip().title()

    stem = Path(source).stem if source else 'document'
    return re.sub(r'[_\-]+', ' ', stem).strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b: Document-type detection
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_KEYWORDS: dict[str, set[str]] = {
    'hr': {
        'employee', 'leave', 'salary', 'benefits', 'recruitment',
        'payroll', 'performance', 'onboarding', 'termination', 'vacation',
    },
    'finance': {
        'revenue', 'expense', 'budget', 'invoice', 'balance sheet',
        'profit', 'loss', 'tax', 'ledger', 'forecast', 'audit', 'fiscal',
    },
    'legal': {
        'contract', 'agreement', 'clause', 'liability', 'jurisdiction',
        'indemnity', 'warranty', 'parties', 'obligations', 'breach',
    },
    'technical': {
        'api', 'function', 'method', 'class', 'database', 'server',
        'deployment', 'architecture', 'endpoint', 'configuration',
    },
    'policy': {
        'policy', 'procedure', 'compliance', 'regulation', 'guideline',
        'standard', 'requirement', 'mandate', 'enforcement',
    },
    'sales': {
        'customer', 'pipeline', 'quota', 'deal', 'prospect', 'churn',
        'lead', 'crm', 'opportunity', 'revenue', 'conversion',
    },
}


def detect_doc_type(text: str, kb: str) -> str:
    """Infer document domain from KB name (most reliable) or keyword density.

    Returns a short label like 'hr', 'finance', 'technical', or '' if
    nothing is clear enough.
    """
    kb_lower = (kb or '').lower()
    if kb_lower in _TYPE_KEYWORDS:
        return kb_lower

    sample = text[:1500].lower()
    scores = {t: sum(1 for kw in kws if kw in sample)
              for t, kws in _TYPE_KEYWORDS.items()}
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score >= 3 else ''


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Section-marker injection
# ─────────────────────────────────────────────────────────────────────────────
# RecursiveCharacterTextSplitter is configured with "\n\n===" as its first
# separator, so every "=== SECTION: ... ===" we inject here becomes a
# guaranteed chunk boundary — no chunk will ever span two logical sections.
#
# Pattern priority:
#   1. Markdown headings  (# H1, ## H2, ### H3)
#   2. Underlined headings (Title\n====  and  Title\n----)
#   3. Numbered sections  (1. Title  or  2.3 Sub-title)
#   4. ALL-CAPS headings  (only enabled when other section types are present,
#      to avoid false positives in documents that just use capitals normally)

_DETECT_SECTION_PATTERNS = [
    re.compile(r'^#{1,3}\s+(.+)',               re.MULTILINE),
    re.compile(r'^([^\n]{4,})\n[=]{3,}\s*$',   re.MULTILINE),
    re.compile(r'^([^\n]{4,})\n[-]{3,}\s*$',   re.MULTILINE),
    re.compile(r'^\d+\.\d*\s+([A-Z].{2,60})$', re.MULTILINE),
    re.compile(r'^([A-Z][A-Z0-9 \-\/]{4,50}):?\s*$', re.MULTILINE),
]

_CAPS_HEADER_RE = re.compile(r'^([A-Z][A-Z0-9 \-\/]{4,50}):?\s*$', re.MULTILINE)


def _has_structural_headers(text: str) -> bool:
    """Return True if the text contains at least one structural header."""
    return any(p.search(text) for p in _DETECT_SECTION_PATTERNS[:4])


def _caps_header_count(text: str) -> int:
    """Count ALL-CAPS heading candidates in the text."""
    return len(_CAPS_HEADER_RE.findall(text))


def inject_section_markers(text: str) -> str:
    """Replace recognised section headers with normalised === SECTION === lines.

    Each injected marker produces a blank-line-separated block that
    RecursiveCharacterTextSplitter splits on before resorting to paragraph
    or sentence boundaries.
    """
    # 1. Markdown H1–H3
    text = re.sub(
        r'^(#{1,3})\s+(.+)$',
        lambda m: f'\n\n=== SECTION: {m.group(2).strip()} ===\n',
        text, flags=re.MULTILINE,
    )

    # 2. Underlined headings with ====
    text = re.sub(
        r'^([^\n]{4,})\n[=]{3,}\s*$',
        lambda m: f'\n\n=== SECTION: {m.group(1).strip()} ===\n',
        text, flags=re.MULTILINE,
    )

    # 3. Underlined headings with ----
    text = re.sub(
        r'^([^\n]{4,})\n[-]{3,}\s*$',
        lambda m: f'\n\n=== SECTION: {m.group(1).strip()} ===\n',
        text, flags=re.MULTILINE,
    )

    # 4. Numbered sections: "1. Title" or "2.3 Sub-title"
    text = re.sub(
        r'^(\d+\.\d*\s+)([A-Z].{2,60})$',
        lambda m: f'\n\n=== SECTION: {m.group(1).strip()} {m.group(2).strip()} ===\n',
        text, flags=re.MULTILINE,
    )

    # 5. ALL-CAPS headings — inject when ≥2 are present (reduces false positives
    #    from documents that occasionally capitalise words) or when other
    #    structural header types co-exist in the same document.
    if _has_structural_headers(text) or _caps_header_count(text) >= 2:
        text = re.sub(
            r'^([A-Z][A-Z0-9 \-\/]{4,50}):?\s*$',
            lambda m: f'\n\n=== SECTION: {m.group(1).strip().title()} ===\n',
            text, flags=re.MULTILINE,
        )

    # Collapse any triple+ newlines introduced by substitutions
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b: Document-level context header
# ─────────────────────────────────────────────────────────────────────────────

def _build_doc_header(title: str, kb: str, doc_type: str, source: str) -> str:
    """One-line === DOCUMENT === header prepended before the document body.

    This becomes the first chunk's opening line, giving the splitter a
    natural document-start boundary and the retriever an anchor chunk that
    describes the whole document.
    """
    parts = [f'DOCUMENT: {title}']
    if kb:       parts.append(f'KB: {kb}')
    if doc_type: parts.append(f'TYPE: {doc_type}')
    if source:   parts.append(f'FILE: {Path(source).name}')
    return '=== ' + ' | '.join(parts) + ' ===\n\n'


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_document(doc: Document) -> Document:
    """Clean, structure, and enrich a single raw document.

    Returns a new Document with:
      - Cleaned page_content (artifacts removed, whitespace normalised)
      - === SECTION === markers injected at detected header positions
      - === DOCUMENT === header prepended for document-level context
      - Enriched metadata: title, doc_type, section_count, word_count
    """
    source   = doc.metadata.get('source', '')
    kb       = doc.metadata.get('kb', '')
    page     = doc.metadata.get('page', None)   # set by PyPDFLoader
    raw_text = doc.page_content

    # Stage 1 — clean
    text = clean_text(raw_text)
    if not text:
        logger.debug("Empty document after cleaning: %s", source)
        return doc

    # Stage 2 — detect title and doc type
    title    = extract_title(text, source, page)
    doc_type = detect_doc_type(text, kb)

    # Stage 3 — inject section markers
    text = inject_section_markers(text)

    # Count before adding the document header
    section_count = text.count('=== SECTION:')
    word_count    = len(raw_text.split())

    # Prepend document-level header
    header = _build_doc_header(title, kb, doc_type, source)
    text   = header + text

    new_meta = {
        **doc.metadata,
        'title':         title,
        'doc_type':      doc_type,
        'section_count': section_count,
        'word_count':    word_count,
    }

    logger.debug(
        "Preprocessed '%s' (page=%s): %d words, %d sections, type=%r",
        Path(source).name if source else 'unknown',
        page, word_count, section_count, doc_type or 'generic',
    )
    return Document(page_content=text, metadata=new_meta)


def preprocess_documents(docs: list[Document]) -> list[Document]:
    """Preprocess all documents; empty docs are returned unchanged."""
    processed = [preprocess_document(d) for d in docs]
    total_words = sum(d.metadata.get('word_count', 0) for d in processed)
    logger.info(
        "Preprocessing complete — %d doc(s), %d total words, %d section(s) detected",
        len(processed),
        total_words,
        sum(d.metadata.get('section_count', 0) for d in processed),
    )
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Post-split contextualisation
# ─────────────────────────────────────────────────────────────────────────────
# After RecursiveCharacterTextSplitter runs, chunks that fall in the middle of
# a long section have lost their section context.  This pass adds a compact
# one-line prefix "[Document | KB | Section]" to every chunk so that each
# retrieved snippet is fully self-contained.
#
# We track the "current section" per source document by scanning chunks in
# order.  A chunk that opens with === SECTION: X === updates the tracker;
# subsequent continuation chunks inherit X.  === DOCUMENT === chunks reset
# the tracker.

_SECTION_HEADER_RE  = re.compile(r'^===\s*SECTION:\s*(.+?)\s*===', re.IGNORECASE)
_DOCUMENT_HEADER_RE = re.compile(r'^===\s*DOCUMENT:', re.IGNORECASE)


def contextualise_chunks(chunks: list[Document]) -> list[Document]:
    """Prefix every chunk with a compact context line.

    For each chunk the prefix encodes:
      - Document title  (from metadata['title'])
      - Knowledge base  (from metadata['kb'])
      - Current section (tracked across chunks from same source file)

    Example prefix:
      [Employee Handbook | KB:hr | §Annual Leave]

    This implements the "contextual retrieval" pattern: even a mid-section
    continuation chunk carries enough context for the LLM to know exactly
    what it is reading.
    """
    # source path → name of the section currently being chunked
    current_section: dict[str, str] = {}
    result: list[Document] = []

    for chunk in chunks:
        source  = chunk.metadata.get('source', '')
        content = chunk.page_content.lstrip()

        # Update section tracker from this chunk's opening line
        sec_match = _SECTION_HEADER_RE.match(content)
        doc_match = _DOCUMENT_HEADER_RE.match(content)

        if sec_match:
            current_section[source] = sec_match.group(1).strip()
        elif doc_match:
            current_section[source] = ''   # document header resets section

        # Build context prefix from metadata + tracker
        title    = chunk.metadata.get('title', '')
        kb       = chunk.metadata.get('kb', '')
        section  = current_section.get(source, '')
        doc_type = chunk.metadata.get('doc_type', '')

        parts: list[str] = []
        if title:    parts.append(title)
        if kb:       parts.append(f'KB:{kb}')
        if doc_type and doc_type != kb: parts.append(f'type:{doc_type}')
        if section:  parts.append(f'§{section}')

        prefix = ('[' + ' | '.join(parts) + ']\n') if parts else ''

        result.append(Document(
            page_content=prefix + chunk.page_content,
            metadata=chunk.metadata,
        ))

    logger.info(
        "Contextualised %d chunk(s) across %d source(s)",
        len(result),
        len({c.metadata.get('source', '') for c in result}),
    )
    return result
