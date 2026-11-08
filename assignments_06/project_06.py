"""Week 6 mini-project: Groundwork Coffee Q&A assistant."""

from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- Step 1: Setup ---

docs_dir = Path("resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Docs directory found: {docs_dir}")

# --- Step 2: Load the Documents ---

print("\nLoading documents from:", docs_dir)
documents = SimpleDirectoryReader(str(docs_dir)).load_data()
print(f"Loaded {len(documents)} document(s).")
print("File names:")
for doc in documents:
    file_name = doc.metadata.get("file_name", "unknown")
    print(f"  - {file_name}")

# --- Step 3: Build the Index and Query Engine ---

print("\nBuilding index (this may take a moment)...")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant ---

print("\n" + "=" * 70)

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print("=" * 70)
    print(f"Question: {q}\n")
    resp = query_engine.query(q)
    print(f"Answer: {resp}\n")

    for i, node in enumerate(resp.source_nodes, 1):
        name = node.node.metadata.get("file_name", "unknown")
        print(f"  [{i}] {name}  score={node.score:.4f}")

    top = resp.source_nodes[0]
    doc_name = top.node.metadata.get("file_name", "unknown")
    score = top.score
    snippet = top.node.get_content()[:200]
    print(f"Top source: {doc_name}  (score={score:.4f})")
    print(f"Snippet: {snippet}...")
    print()

# Q (Step 4 reflection): Did the assistant sound confident and accurate?
# A: All five answers were correct and confident. But the retrieval story is messier.
#    Q4 (0.77) and Q5 (0.62) had a clean top-1 hit on the right file.
#    Q1 (hours): top-1 was our_story.txt (0.51) -- WRONG file. The correct faq.txt
#       was only at [3] with score 0.47.
#    Q2 (dairy-free): top-1 was seasonal_specials.txt (0.44). The correct menu.txt
#       was at [2] with score 0.35.
#    Q3 (loyalty): top-1 was faq.txt but with a very low score (0.28) -- the top
#       of that chunk happens to be the hours section, not the loyalty section.
#    In every case the correct file was still inside the top-3, so top_k=3 saved us.
#    With top_k=1, Q1, Q2 and Q3 would likely have been wrong. Semantic search
#    on small documents is fragile -- everything sits close to everything else
#    in the embedding space.

# --- Step 5: Find a Failure ---

print("\n" + "=" * 70)
print("--- Step 5: Finding a failure case ---\n")

failure_query = "What is the name of Groundwork's head roaster and how long have they worked there?"

print(f"Question: {failure_query}\n")
fail_resp = query_engine.query(failure_query)
print(f"Answer: {fail_resp}\n")
print("All retrieved sources:")
for i, node in enumerate(fail_resp.source_nodes, 1):
    name = node.node.metadata.get("file_name", "unknown")
    snippet = node.node.get_content()[:200]
    print(f"  [{i}] {name}  score={node.score:.4f}")
    print(f"      {snippet}...")
print()

# Q (Step 5 reflection):
# Q: What went wrong -- wrong retrieval, missing information, the model guessed anyway?
# A: None of those. The information is genuinely not in the documents -- there is no
#    "head roaster" mentioned anywhere, only the two founders. Retrieval returned three
#    chunks with low scores (0.63, 0.52, 0.43), none of which mention a head roaster.
#    The model recognized this and refused to answer instead of guessing. So the
#    retrieval "failed" to find relevant chunks (because there were none), but the
#    system as a whole handled it correctly.
#
# Q: Did the model's tone change -- did it become less certain, or did it still sound
#    confident even when it was wrong?
# A: The tone stayed confident, but not in the sense of "confidently wrong". It was
#    confident about the FACT that the info is missing: "The provided information does
#    not include the name..." That is the right kind of confidence. The model was not
#    hedging with "I'm not sure" or "maybe" -- it clearly stated what the context did
#    and did not contain. This is good RAG behavior: ground the answer in what was
#    retrieved, and say so when nothing relevant was retrieved.
#
# Q: What would you change about the system to improve it?
# A: Two things. (1) Add a score threshold -- if the top retrieved score is below ~0.65,
#    skip the LLM call and tell the user "I could not find this in the documents", which
#    saves tokens and is more explicit. (2) Since the model's answer was correct here,
#    the bigger win would be on queries where the wrong chunk scores just above the
#    threshold -- then the model might answer from the wrong chunk. A reranker
#    (cross-encoder) after retrieval would help pick the truly best chunk.

# --- Step 6: Reflection ---
#
# Q1: How many lines of code did the LlamaIndex implementation take, and what does that
#     tell you about the value of using a framework?
# A: The LlamaIndex core (Steps 1-3) was about 20 lines: load docs, set embed model,
#     set LLM, build VectorStoreIndex, create query engine. The keyword RAG lesson
#     needed a custom retrieval function of ~40 lines PLUS manual chunking, embedding,
#     and index-building code. Doing those three steps from scratch (chunking +
#     embedding with OpenAI + storing in a vector structure + cosine similarity search)
#     would be 80-150 lines. LlamaIndex packs all of that into three method calls.
#     The value is fewer places for bugs, a stable interface, and the ability
#     to swap vector stores or embedding models without rewriting everything.
#
# Q2: Describe a different use case (not a coffee shop) where RAG would add value.
# A: Internal HR policy assistant. A company has hundreds of pages of HR policies
#     (leave, benefits, performance reviews, code of conduct) that change yearly.
#     Employees ask natural-language questions in Slack: "How many sick days do I
#     have?" or "Can I work from Portugal for a month?". A keyword search on the
#     intranet misses synonyms; an ungrounded LLM hallucinates policy details that
#     could get the company sued. RAG retrieves the exact policy section and grounds
#     the answer, with source links so employees can verify. Updates to policies
#     just mean reindexing, not retraining.
#
# Q3: What is one failure mode RAG cannot fully prevent, even when retrieval works?
# A: Hallucination when the retrieved chunk is relevant but does not fully answer the
#     question. Example: the question asks "when does the loyalty program expire?" and
#     retrieval returns the chunk that describes how the program works (points, redeem,
#     sign-up) -- very high score -- but that chunk does not mention an expiration date.
#     The LLM sees a relevant chunk, assumes it has what it needs, and fills the gap
#     with a plausible-sounding answer ("points expire after 12 months"). Retrieval
#     "worked" in the sense of finding the right topic, but the model still fabricated
#     the specific fact. This is why faithfulness evaluation and human review remain
#     important even with good retrieval.
