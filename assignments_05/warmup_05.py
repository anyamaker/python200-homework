"""Week 5 warmup: practicing the OpenAI Chat Completions API."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --- Chat Completions API ---
# API Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response text:", response.choices[0].message.content)
print("Model used:", response.model)
print("Total tokens:", response.usage.total_tokens)

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"\n--- Temperature {temp} ---")
    print(response.choices[0].message.content)

# Q: What do you notice about how the outputs differ? Which temperature for consistent output?
# A: At 0 the answer is the same every time (deterministic). At 0.7 it varies but stays on topic.
#    At 1.5 it gets wilder and may go off-topic or pick strange names. For reproducible output use 0.

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0,
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Option {i}: {choice.message.content}")

# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)

print("\n--- With max_tokens=15 ---")
print(response.choices[0].message.content)

# Q: What happened, and why use max_tokens in a real app?
# A: The response was cut off mid-sentence at 15 tokens. max_tokens caps cost and response time,
#    useful when you only need a short answer or want to protect a budget.

# --- System Messages and Personas ---
# System Q1

# Personality 1: patient Python tutor
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\n--- Python Tutor persona ---")
print(response.choices[0].message.content)

# Personality 2: grumpy drill sergeant
messages = [
    {"role": "system", "content": "You are a grumpy drill sergeant. You explain things bluntly, with no patience for excuses, but you still give clear answers."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\n--- Drill Sergeant persona ---")
print(response.choices[0].message.content)

# Q: What changed?
# A: The tutor persona is warm and encouraging; the sergeant persona is blunt and direct.
#    Same question, same information, but the tone and word choice are very different.

# System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\n--- Stateless memory demo ---")
print(response.choices[0].message.content)

# Q: Why does the model know Jordan's name, even though it's stateless?
# A: Because we sent the whole conversation history in the messages list.
#    The model sees the earlier user message where Jordan introduced themselves,
#    so it can reference that in the reply. Nothing is stored between calls.

# --- Prompt Engineering ---
# Prompt Q1 — Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

print("\n--- Zero-Shot Sentiment ---")
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify the sentiment of this review as positive, negative, or mixed:\n\n{review}"}],
    )
    print(f"Review {i}: {response.choices[0].message.content.strip()}")

# Prompt Q2 — One-Shot

print("\n--- One-Shot Sentiment ---")
for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
Sentiment:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"Review {i}: {response.choices[0].message.content.strip()}")

# Q: Did adding one example change the format or consistency compared to Q1?
# A: Yes -- with one example the model copies the format and returns just the label
#    (e.g. "positive") instead of a full sentence. One-shot controls the output shape.

# Prompt Q3 — Few-Shot

print("\n--- Few-Shot Sentiment ---")
for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of each review as positive, negative, or mixed.

Example 1:
Review: "Absolutely love this product, works perfectly!"
Sentiment: positive

Example 2:
Review: "Terrible experience, would not recommend."
Sentiment: negative

Example 3:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
Sentiment:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"Review {i}: {response.choices[0].message.content.strip()}")

# Q: Compare zero-shot, one-shot, few-shot. When would you choose each?
# A: Zero-shot: fast and cheap, good when the task is simple and the model already knows it.
#    One-shot: when you need a specific output format and one example is enough to show it.
#    Few-shot: when the task is nuanced or you need to cover several categories (like here,
#    three classes) so the model sees the full range of expected outputs.

# Prompt Q4 — Chain of Thought

prompt = """Show your reasoning step by step, then give the final answer on its own line labelled 'Final answer: <value>'.

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("\n--- Chain of Thought ---")
print(response.choices[0].message.content)

# Q: Why does asking the model to reason step by step improve accuracy?
# A: Breaking the problem into steps makes the model show its work, so it does not skip
#    a calculation or jump to a wrong answer. Each step is a small, checkable operation,
#    which is easier for the model to get right than one big guess.

# Prompt Q5 — Structured Output

import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""Analyze the sentiment of this review and respond ONLY with valid JSON.
Keys: sentiment (positive/negative/mixed), confidence (float between 0 and 1), reason (one short sentence).

Review: "{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)
raw = response.choices[0].message.content
print("\n--- Structured Output ---")
print("Raw response:", raw)

try:
    data = json.loads(raw)
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except json.JSONDecodeError:
    print("ERROR: response was not valid JSON. Raw response above.")

# Q: Why ask for JSON, and why wrap parsing in try/except?
# A: JSON is easy to parse in code, so the result can be used by other functions.
#    Models sometimes add markdown or extra text around the JSON, which breaks parsing.
#    response_format={"type": "json_object"} forces clean JSON, and try/except catches
#    any remaining problem and shows the raw response for debugging.

# Prompt Q6 — Delimiters

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("\n--- Delimiters (instructions) ---")
print(response.choices[0].message.content)

# Second prompt: text that is NOT a set of instructions
non_instructions = "The quick brown fox jumps over the lazy dog on a sunny afternoon."

prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instructions}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("\n--- Delimiters (non-instructions) ---")
print(response.choices[0].message.content)

# Q: What problem do delimiters help prevent?
# A: Without delimiters the model can confuse the user's text with the instructions,
#    especially if the text contains phrases like "ignore previous instructions" (prompt injection).
#    Delimiters clearly mark where data starts and ends, so the model knows what to act on.

# --- Local Models with Ollama ---
# Ollama Q1

# Ollama output (qwen3:0.6b, final answer only):
ollama_output = """
A large language model is an artificial intelligence system trained on vast amounts
of text, allowing it to understand and generate human-like language. These models
process massive datasets to learn patterns, enabling them to understand context,
perform tasks like writing, and create content that mimics human thought.
"""
print("\n--- Ollama (qwen3:0.6b) ---")
print(ollama_output)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}],
)
print("\n--- OpenAI (gpt-4o-mini) ---")
print(response.choices[0].message.content)

# Q: What differences did you notice? One advantage and one disadvantage of local models?
# A: Both fit in two sentences, but the small local model uses looser wording
#    ("mimics human thought", "understand context") while gpt-4o-mini is more precise.
#    The local model also shows a long visible "thinking" phase before the answer.
#    Advantage of local: free, private, offline. Disadvantage: weaker instruction-following
#    and less polished wording; needs disk space and a decent machine.