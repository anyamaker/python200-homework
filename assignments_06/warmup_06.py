"""Week 6 warmup: RAG concepts, keyword RAG, semantic RAG, LlamaIndex."""

from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---
# Concepts Q1

# Scenario A: legal team, hundreds of PDFs updated quarterly -> RAG.
# The documents change often and are too many to fit in a prompt. RAG retrieves
# the right pages at query time, so updates only require reindexing, not retraining.

# Scenario B: startup, specific brand voice, 3,000 in-house examples -> fine-tuning.
# The style is not in the model's training data, but the startup has plenty of
# examples. Fine-tuning bakes that voice into the model itself.

# Scenario C: data analyst, one two-page report, one-off question -> prompt engineering.
# The report is tiny and fits in a prompt. No indexing or training needed.
# Just paste the report into the prompt with the question.

# Concepts Q2

# A confident wrong answer is more harmful than an uncertain one because people act on
# confidence. When a model says "I am not sure," the user checks, asks someone, or looks it up.
# When it says the wrong thing in a confident tone, the user trusts it and moves on.

# Real example: a lawyer submits a court filing that cites cases invented by the model.
# The model sounded certain, the lawyer did not verify, and the case was sanctioned in court.
# Another: a doctor uses an AI suggestion for a drug dosage that is confidently wrong.

# Tone matters because humans read certainty as a signal of accuracy. The same wrong fact
# delivered as a guess gets checked. Delivered as a fact, it slips through.

# Concepts Q3
#
# Correct order (preparation first, then answering):
#
# 1. Extract text from source documents
#    Read the raw PDFs/text files and pull out plain text.
# 2. Split text into chunks
#    Cut long documents into smaller pieces so retrieval can return just the relevant part.
# 3. Convert text chunks into embeddings
#    Turn each chunk into a vector of numbers that captures its meaning.
#    (The vectors are then stored in an index, which is implied between 3 and 4.)
# 4. Receive the user's query
#    The user types a question.
# 5. Embed the user's query
#    Turn the question into a vector using the same embedding model.
# 6. Retrieve the most relevant chunks
#    Find the chunks whose vectors are closest to the query vector (cosine similarity).
# 7. Inject retrieved chunks into the prompt
#    Build a prompt that includes the retrieved text plus the user's question.
# 8. Generate a response from the LLM
#    Send the augmented prompt to the model and return its answer.

# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

print("\n--- Keyword Q1 ---")
result = simple_keyword_retrieval(query, documents, verbose=True)
print("\nSelected document:", result[0][0])

# Q: Which document was selected and why?
# A: loyalty.txt was selected, but that is WRONG. Two problems combined:
#    1. The word "hours" does not actually appear in hours.txt ("open 7am to 7pm" instead),
#       so "hours" matched nothing.
#    2. "your" is not in the stopword list, so it falsely matched hiring.txt and loyalty.txt.
#    Three documents tied at score 1, and the sort picked loyalty.txt by name.
#    This shows how brittle keyword RAG is: it matches exact words, not meaning.

# Keyword Q2

query2 = "Do you have anything without caffeine?"

print("\n--- Keyword Q2 ---")
result2 = simple_keyword_retrieval(query2, documents, verbose=True)
print("\nSelected document:", result2[0][0])

# Q: Which document was selected, and did keyword RAG get it right?
# A: No document was selected -- the function returned "None found" because
#    none of the query words (anything, caffeine, do, have, without) appear in any
#    of the four documents. But the CORRECT answer is menu.txt: it lists espresso,
#    lattes, cold brew, oat milk, almond milk -- exactly the drinks you would look at
#    to decide about caffeine. Keyword RAG misses it because it matches words, not meaning.
#
# Q: What kind of retrieval would do better?
# A: Semantic RAG. An embedding model would place "without caffeine" near the menu
#    chunk, because "espresso" and "caffeine" are semantically related even without
#    sharing a word.

# Keyword Q3
#
# Prediction (write BEFORE running):
# Query: "How do I sign up for rewards?"
# After filtering, the meaningful words are: sign, rewards. (how, do, i, for are stopwords
# or short filler words -- "how", "for", "do" are in the stopword list; "i" is not but
# is very short and not in any doc; "sign" and "rewards" are the real signals.)
# loyalty.txt says "Join our loyalty program... earn one point per dollar spent...
# Redeem 100 points for a free drink of your choice." It does not literally contain
# "sign up" or "rewards", but it does contain "loyalty" and "program".
# keyword RAG only looks for exact words, so I predict it will fail again --
# either no match at all, or a wrong tie-break.

query3 = "How do I sign up for rewards?"

print("\n--- Keyword Q3 ---")
result3 = simple_keyword_retrieval(query3, documents, verbose=True)
print("\nSelected document:", result3[0][0])

# Post-run: prediction was correct. None of the query words matched any document.
# loyalty.txt is the right answer (it describes the loyalty program you sign up for),
# but it uses "loyalty program" and "Redeem points", not "sign up" or "rewards".
# This is the third failure of keyword RAG on the same four documents -- exactly
# the limitation the lesson describes. Semantic search would handle all three.

# --- Semantic RAG Concepts ---
# Semantic Q1

# Q: What is a vector embedding?
# A: An embedding is a list of numbers (a vector) that represents the meaning of a piece
#    of text. Texts with similar meaning get vectors that are close together in that space.

# Q: Two chunks have cosine similarity 0.85 and 0.30 with the query. Which is more relevant?
# A: 0.85 is more relevant. Cosine similarity goes from -1 to 1; values near 1 mean the
#    vectors point in nearly the same direction, i.e. the meanings are close.
#    0.85 means strongly related; 0.30 means only weakly related.

# Q: Why can semantic search find a relevant chunk even when no exact query words appear?
# A: Because embeddings capture meaning, not spelling. "Car" and "automobile" produce
#    similar vectors even though they share no letters. The query "without caffeine"
#    lands near a menu chunk that says "espresso, oat milk" because those concepts
#    co-occur in meaning-space.

# Semantic Q2
#
# | Feature                 | Keyword RAG                     | Semantic RAG                                    |
# |-------------------------|---------------------------------|-------------------------------------------------|
# | What is compared?       | Exact word overlap              | Vector embeddings (meaning)                     |
# | What is retrieved?      | Full document                   | Top-k most similar chunks                       |
# | Can it handle synonyms? | No                              | Yes                                             |
# | Storage format          | Plain text dictionary           | Vector index / vector store                     |
# | Relevance score         | Number of overlapping keywords  | Cosine similarity (between -1 and 1)            |

# --- LlamaIndex ---

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Q1

print("\n--- LlamaIndex Q1 ---")
print("Loading Brightleaf PDFs and building index...")

docs = SimpleDirectoryReader("resources/brightleaf_pdfs").load_data()
print(f"Loaded {len(docs)} pages from Brightleaf PDFs.")

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built. Ready to answer.\n")

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print("=" * 70)
    print(f"Question: {q}")
    response = query_engine.query(q)
    print(f"Answer: {response}\n")
    print("Retrieved sources:")
    for i, node_with_score in enumerate(response.source_nodes, 1):
        score = node_with_score.score
        text = node_with_score.node.get_content()[:150]
        print(f"  [{i}] score={score:.4f}")
        print(f"      {text}...")
    print()

# Q: Do the retrieved chunks look relevant? Is the tone confident or hedging?
# A: Source [1] for each question is clearly relevant (score 0.77 for benefits, 0.68 for
#    security), but sources [2] and [3] are noise (partnerships doc, mission doc).
#    The model's answer is confident and specific -- no "based on the context" hedging.
#    It ignored the noisy chunks and answered from the right one.
#
# Unexpected?: Did anything unexpected get retrieved?
# A: Yes -- sources [2] and [3] for both questions are the same two documents
#    (partnerships / mission), which have nothing to do with the query. This is because
#    similarity_top_k=3 always returns three chunks, even when only one is truly relevant.

# LlamaIndex Q2

print("\n--- LlamaIndex Q2 ---")
q2 = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    print(f"\n=== similarity_top_k = {k} ===")
    engine = index.as_query_engine(similarity_top_k=k)
    resp = engine.query(q2)
    print(f"Answer: {resp}\n")
    print(f"Sources ({len(resp.source_nodes)}):")
    for i, node_with_score in enumerate(resp.source_nodes, 1):
        print(f"  [{i}] score={node_with_score.score:.4f}")
        print(f"      {node_with_score.node.get_content()[:100]}...")

# Q2: How did the response change? Is more retrieved context always better?
# A: The answer is essentially identical for k=1 and k=5. k=1 uses one relevant chunk
#    (score 0.77); k=5 adds four more at scores 0.59, 0.57, 0.48, 0.44 -- none about
#    benefits. More context is not always better: the extra chunks are noise, they cost
#    more tokens, and they don't improve the answer.

# LlamaIndex Q3

print("\n--- LlamaIndex Q3 ---")

# Query that should be hard: answer is NOT in the documents.
hard_query = "What is BrightLeaf's stock ticker symbol?"

print(f"Question: {hard_query}\n")
hard_resp = query_engine.query(hard_query)
print(f"Answer: {hard_resp}\n")
print(f"All retrieved sources ({len(hard_resp.source_nodes)}):")
for i, node_with_score in enumerate(hard_resp.source_nodes, 1):
    print(f"  [{i}] score={node_with_score.score:.4f}")
    print(f"      {node_with_score.node.get_content()[:150]}...")

# Q3: What did you expect, what happened, what would you change?
# A: I expected either a hallucinated ticker or a hedged non-answer. Instead the model
#    correctly said the information is not in the context -- good behavior. All three
#    retrieval scores are low (0.58, 0.55, 0.54) and close together, which signals that
#    nothing truly relevant was found. To improve: add a score threshold, so if the top
#    score is below ~0.65 we skip the LLM call and tell the user "not in the documents".

# LlamaIndex Q4

print("\n--- LlamaIndex Q4 ---")

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Judge LLM (separate from the reader -- it grades the reader's answer)
judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

faithfulness_eval = FaithfulnessEvaluator(llm=judge_llm)
relevancy_eval = RelevancyEvaluator(llm=judge_llm)


def evaluate_query(question):
    """Run one query and grade the answer with both evaluators."""
    print("=" * 70)
    print(f"Question: {question}")
    resp = query_engine.query(question)
    print(f"Answer: {resp}\n")

    faith_result = faithfulness_eval.evaluate_response(query=question, response=resp)
    relev_result = relevancy_eval.evaluate_response(query=question, response=resp)

    print(f"Faithfulness score: {faith_result.score}  (passing={faith_result.passing})")
    print(f"Relevancy score:    {relev_result.score}  (passing={relev_result.passing})")
    print()


# Query 1: expected high quality
evaluate_query("What employee benefits does BrightLeaf offer?")

# Query 2: expected low quality (not in the documents)
evaluate_query("What is the CEO's favorite color?")

# Q4 comment block:
# Q: What does a faithfulness score of 1.0 mean? What would 0.0 indicate?
# A: 1.0 means the answer is fully supported by the retrieved chunks -- no hallucination.
#    0.0 means the judge found the answer is NOT supported by the context. Note that 0.0
#    can also happen when the model correctly says "I don't know" -- the judge treats
#    a refusal as "not supported", which is a limitation of this metric.
#
# Q: What does relevancy measure, and how is it different from faithfulness?
# A: Relevancy asks "does the answer address the question?" Faithfulness asks "is the
#    answer grounded in the retrieved context?" They are independent: an answer can be
#    faithful (quotes the context) but irrelevant (off-topic), or relevant (on-topic)
#    but unfaithful (hallucinated). Both need checking.
#
# Q: Did the scores change between the two queries? Why?
# A: Yes. Benefits: 1.0 / 1.0 (the answer is grounded and on-topic). CEO color: 0.0 / 0.0.
#    This second case looks "bad" but the model actually behaved correctly -- it said
#    "not in the documents". Both judges score a refusal as 0, because it is neither
#    grounded in a chunk nor answering the question. So a 0 / 0 here is not proof of
#    bad behavior -- it means the answer was "no information", which is honest.
#
# Q: What is LLM-as-a-judge, and why is it used for RAG evaluation?
# A: LLM-as-a-judge means using a second LLM to grade the first LLM's output. RAG answers
#    are free-form text -- there is no single correct string to compare against like in
#    classification. A judge LLM reads the question, the retrieved context, and the answer,
#    and decides whether the answer is grounded and relevant. It scales much better than
#    human graders and is cheaper. Its weakness: it cannot always distinguish a good
#    refusal ("I don't know") from a bad hallucination.
