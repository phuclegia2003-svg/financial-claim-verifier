
import os
import io
import json
import re
import time

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL = "gemini-3.8-flash"

st.set_page_config(
    page_title="Financial Claim Verifier",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# GEMINI CLIENT
# ---------------------------------------------------------

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "GOOGLE_API_KEY is not set. "
        "Please set it in PowerShell before running the app."
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ---------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes):
    """
    Read and chunk the annual report.

    Streamlit caches this function using the PDF bytes as the key.
    Therefore, pressing the Verify button will NOT re-read the PDF
    unless the uploaded report actually changes.
    """

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text
        })

    chunks = []

    chunk_size = 1200

    for page in pages:

        text = page["text"]

        if not text.strip():
            continue

        for start in range(0, len(text), chunk_size):

            chunk_text = text[start:start + chunk_size]

            chunks.append({
                "page": page["page"],
                "text": chunk_text
            })

    # Build TF-IDF index once instead of rebuilding it for every claim.
    documents = [item["text"] for item in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    return pages, chunks, vectorizer, matrix


def retrieve_evidence(query, chunks, vectorizer, matrix, top_k=5):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    evidence = []

    for index in top_indices:

        evidence.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(similarities[index])
        })

    return evidence


# ---------------------------------------------------------
# GEMINI HELPERS
# ---------------------------------------------------------

def clean_json_response(text):

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def extract_claims(answer):

    prompt = f"""
You are a financial claim extraction system.

Break the following AI-generated financial answer
into individual factual claims.

Rules:
- Separate compound statements.
- Keep numerical claims separate.
- Keep causal claims separate.
- Do not add new information.
- Do not judge whether claims are true.
- Return ONLY valid JSON.

Answer:
{answer}

Return:

[
    {{"claim": "individual claim 1"}},
    {{"claim": "individual claim 2"}}
]
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = clean_json_response(response.text)

    return json.loads(text)


def verify_single_claim(claim, evidence):

    evidence_text = ""

    for i, item in enumerate(evidence, start=1):

        evidence_text += f"""
EVIDENCE {i}
PAGE: {item['page']}
{item['text']}
-------------------------
"""

    prompt = f"""
You are a financial document verification system.

Determine whether the following financial claim is supported
by the supplied annual-report evidence.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

Use exactly one classification:

SUPPORTED
The evidence directly supports the claim.

CONTRADICTED
The evidence directly conflicts with the claim.

UNSUPPORTED
The evidence does not establish the claim.

Rules:
1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Check numerical claims carefully.
4. Causal claims require evidence of the stated cause.
5. If evidence is insufficient, use UNSUPPORTED.

Return ONLY valid JSON:

{{
    "classification": "SUPPORTED",
    "reason": "brief explanation",
    "page": "page number"
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = clean_json_response(response.text)

    return json.loads(text)


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

st.sidebar.header("1. Upload report")

uploaded_file = st.sidebar.file_uploader(
    "Upload an annual report PDF",
    type=["pdf"]
)


# ---------------------------------------------------------
# LOAD REPORT
# ---------------------------------------------------------

pages = None
chunks = None
vectorizer = None
matrix = None

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Reading annual report..."):

        pages, chunks, vectorizer, matrix = process_pdf(
            pdf_bytes
        )

    st.sidebar.success(
        f"Loaded {len(pages)} pages and "
        f"{len(chunks)} evidence chunks."
    )


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

st.title("Financial Claim Verifier")

st.write(
    "Upload a financial report and enter an AI-generated "
    "financial answer. The system checks individual claims "
    "against evidence from the report."
)


st.header("2. Enter AI-generated financial analysis")

answer = st.text_area(
    "Paste the answer you want to verify:",
    height=220
)


if st.button(
    "Verify Financial Claims",
    type="primary"
):

    if not uploaded_file:

        st.error(
            "Please upload an annual report first."
        )

    elif not answer.strip():

        st.error(
            "Please enter an AI-generated answer."
        )

    else:

        # -------------------------------------------------
        # STEP 1: EXTRACT CLAIMS
        # -------------------------------------------------

        with st.spinner(
            "Extracting individual financial claims..."
        ):

            try:

                claims = extract_claims(answer)

            except Exception as e:

                st.error(
                    "Could not extract claims from Gemini.\n\n"
                    + str(e)
                )

                st.stop()


        if not isinstance(claims, list):

            st.error(
                "Gemini did not return the expected claim list."
            )

            st.stop()


        # -------------------------------------------------
        # STEP 2: RETRIEVE + VERIFY EACH CLAIM
        # -------------------------------------------------

        results = []
        all_evidence = []

        progress = st.progress(0)

        total_claims = len(claims)

        for claim_number, item in enumerate(
            claims,
            start=1
        ):

            claim = item.get(
                "claim",
                ""
            ).strip()

            if not claim:
                continue


            # ---------------------------------------------
            # Retrieve evidence for THIS claim
            # ---------------------------------------------

            with st.spinner(
                f"Finding evidence for claim "
                f"{claim_number}/{total_claims}..."
            ):

                evidence = retrieve_evidence(
                    claim,
                    chunks,
                    vectorizer,
                    matrix,
                    top_k=5
                )


            # ---------------------------------------------
            # Verify THIS claim
            # ---------------------------------------------

            with st.spinner(
                f"Verifying claim "
                f"{claim_number}/{total_claims}..."
            ):

                try:

                    result = verify_single_claim(
                        claim,
                        evidence
                    )

                except Exception as e:

                    result = {
                        "classification": "ERROR",
                        "reason": str(e),
                        "page": "Not identified"
                    }


            result["claim"] = claim

            results.append(result)

            all_evidence.append({
                "claim": claim,
                "evidence": evidence
            })

            progress.progress(
                claim_number / total_claims
            )


        progress.empty()


        # -------------------------------------------------
        # STEP 3: DISPLAY RESULTS
        # -------------------------------------------------

        st.header("Verification Results")


        for result in results:

            classification = result.get(
                "classification",
                "UNKNOWN"
            )

            claim = result.get(
                "claim",
                ""
            )

            reason = result.get(
                "reason",
                ""
            )

            page = result.get(
                "page",
                "Not identified"
            )


            if classification == "SUPPORTED":

                st.success(
                    f"SUPPORTED — {claim}"
                )

            elif classification == "CONTRADICTED":

                st.error(
                    f"CONTRADICTED — {claim}"
                )

            elif classification == "UNSUPPORTED":

                st.warning(
                    f"UNSUPPORTED — {claim}"
                )

            else:

                st.info(
                    f"{classification} — {claim}"
                )


            st.write(
                f"**Reason:** {reason}"
            )

            st.write(
                f"**Source page:** {page}"
            )

            st.divider()


        # -------------------------------------------------
        # STEP 4: SHOW RETRIEVED EVIDENCE
        # -------------------------------------------------

        with st.expander(
            "View retrieved evidence"
        ):

            for item in all_evidence:

                st.write(
                    f"### Claim: {item['claim']}"
                )

                for evidence_item in item["evidence"]:

                    st.write(
                        f"**Page {evidence_item['page']}** "
                        f"(retrieval score: "
                        f"{evidence_item['score']:.3f})"
                    )

                    st.write(
                        evidence_item["text"]
                    )

                    st.divider()
