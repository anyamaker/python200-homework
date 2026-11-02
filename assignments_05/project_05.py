"""Week 5 mini-project: AI-powered job application helper."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7, json_mode=False):
    """Send a messages list to the model and return the assistant's text reply.

    Args:
        messages: List of message dicts.
        model: Model name.
        temperature: Sampling temperature.
        json_mode: If True, force the response to be a JSON object.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": 400,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# --- Task 1: Setup and System Prompt ---

SYSTEM_PROMPT = """You are a job application coach for career changers.

You help people with:
- rewriting resume bullet points
- drafting cover letter openings
- answering general questions about job applications

Rules you must always follow:
1. Stay focused on job application materials. If the user asks about something unrelated,
   politely redirect them back to job applications.
2. Always remind the user to review and edit your output before submitting it anywhere.
   You are a drafting assistant, not a final authority.
3. You do not know the user's specific industry norms, so tell them to use their own
   judgment and check with people in their target field.

Keep your answers practical and concise. Be encouraging but honest.
"""

# Q: Why did I write it this way?
# A: The rules are explicit because vague prompts give vague behavior. Naming the three
#    allowed topics keeps the bot from wandering into unrelated chat. The "review and edit"
#    reminder reduces harm if the user submits AI text without checking. Admitting that the
#    model doesn't know the user's industry prevents overconfident advice.

# --- Task 2: Bullet Point Rewriter ---

import json


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Rewrite weak resume bullets into stronger, results-oriented versions.

    Args:
        bullets: List of original bullet strings.

    Returns:
        List of dicts with "original" and "improved" keys.
    """
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON object with one key, "bullets", whose value is a list.
    Each item in the list should have two keys: "original" and "improved".

    The bullet points are between the <bullets> tags:
    <bullets>
    {bullet_text}
    </bullets>
    """

    messages = [{"role": "user", "content": prompt}]
    raw = get_completion(messages, temperature=0.7, json_mode=True)

    data = json.loads(raw)
    results = data["bullets"]
    for item in results:
        print(f"\nOriginal: {item['original']}")
        print(f"Improved: {item['improved']}")

    return results


# --- Task 3: Cover Letter Generator ---


def generate_cover_letter(job_title: str, background: str) -> str:
    """Write a strong cover letter opening paragraph.

    Args:
        job_title: The role the user is applying for.
        background: A short description of the user's background.

    Returns:
        A cover letter opening paragraph as a string.
    """
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information -- which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)

# --- Task 4: Moderation Check ---


def is_safe(text: str) -> bool:
    """Check whether user input is safe to send to the model.

    Args:
        text: The user's message.

    Returns:
        True if the input is not flagged, False if it is.
    """
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    flagged = result.results[0].flagged
    if flagged:
        print("\nJob Application Helper: That message looks inappropriate. "
              "Please rephrase it and try again.")
        return False
    return True


# Test moderation
safe_input = "I need help rewriting my resume for a data engineer role."
unsafe_input = "I want to hurt someone and destroy their property."

print("\n--- Moderation test (safe) ---")
print("Safe input result:", is_safe(safe_input))

print("\n--- Moderation test (unsafe) ---")
print("Unsafe input result:", is_safe(unsafe_input))

# --- Task 5: The Chatbot Loop ---


def run_chatbot():
    """Run the interactive job application helper chatbot."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            if raw_bullets:
                print("\nJob Application Helper: Here are the improved bullets:")
                rewrite_bullets(raw_bullets)
            else:
                print("\nJob Application Helper: No bullets were provided.")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            print("\nJob Application Helper: Here is a draft opening:")
            print(generate_cover_letter(job_title, background))

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()

# --- Task 6: Ethics Reflection (Option A: comment block) ---
#
# Q1: The bot was trained on text written by and about certain kinds of people.
#     How might this produce biased advice?
# A: The training data skews toward corporate, Western, English-speaking professional norms.
#    That means the bot's idea of a "strong" bullet or "confident" cover letter may favor
#    direct, self-promoting language that works in US tech but can feel wrong in other
#    cultures or industries (e.g. academia, public sector, or non-English markets). It may
#    also assume career paths look linear and that "results" mean numeric metrics, which
#    is not true for teachers, caregivers, or creative roles.
#
# Q2: What could go wrong if a job-seeker submitted the bot's output directly without review?
# A: The bot can invent details -- specific numbers, tools, or achievements the user never
#    mentioned -- because it is optimizing for "compelling", not "accurate". If submitted
#    unchecked, the user could be caught misrepresenting themselves in an interview, which
#    damages trust and can cost them the job. It can also introduce subtle errors in tone
#    (too formal for a startup, too casual for a bank) that the user would have caught.
#
# Q3: One guardrail I would add if deploying this professionally:
# A: A mandatory review step: before the user can copy any output, show a short modal
#    that says "This is a draft. Check every fact before you send it." and require them
#    to click "I have reviewed this" to dismiss it. This makes the reminder impossible
#    to skip, not just a line buried in the chat. I would also add a "which facts are
#    from your input vs. invented by the AI" diff, so users see exactly what to verify.