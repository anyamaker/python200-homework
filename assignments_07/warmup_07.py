"""Week 7 warmup: tool definitions, ReAct loop, multi-tool agent, smolagents."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema describing the function to an LLM
celsius_tool_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a temperature from Celsius to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}

print(celsius_tool_schema)

# Call the function directly
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q2
#
# Prediction (BEFORE running):
# Q: Will run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
# A: No. The only available tool is get_current_time, which returns the current time.
#    The query asks about temperature conversion, so the tool is irrelevant. The model
#    should reason that no tool is needed and answer directly from its own knowledge.
# Q: How many API calls?
# A: One. No tool call means no second call to fetch a final answer.

import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""
    SYSTEM_PROMPT = (
        "You are a simple assistant that can tell the current time. "
        "Use the tool get_current_time whenever a user asks about the time."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print("First response received from model...")
    first_message = first_response.choices[0].message

    messages.append({
        "role": "assistant",
        "content": first_message.content,
        "tool_calls": first_message.tool_calls,
    })

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": tool_result,
            })

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        print("Second response received from model...")
        return second_response.choices[0].message.content or ""
    else:
        print("No tools needed....")

    return first_message.content or ""


result_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Result Q2:", result_q2)

# Was the prediction correct? YES. The model saw only get_current_time as a tool,
# decided it was irrelevant to a temperature question, and answered from its own
# knowledge in one API call. Result: 100°C = 212°F, correct.

# Q3

# Extend the tools list: now both get_current_time and celsius_to_fahrenheit are available.
tools_q3 = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    celsius_tool_schema,   # reuse the schema from Q1
]


def run_agent_q3(user_prompt: str) -> str:
    """ReAct agent with two tools: get_current_time and celsius_to_fahrenheit."""
    SYSTEM_PROMPT = (
        "You are a simple assistant. Use get_current_time for time questions "
        "and celsius_to_fahrenheit for Celsius-to-Fahrenheit conversions."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_q3,
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message

    messages.append({
        "role": "assistant",
        "content": first_message.content,
        "tool_calls": first_message.tool_calls,
    })

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if name == "get_current_time":
                result = get_current_time()
            elif name == "celsius_to_fahrenheit":
                result = celsius_to_fahrenheit(args["celsius"])
            else:
                result = f"Error: unknown tool {name}."

            print(f"Tool called: {name}({args})")
            print(f"Tool result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result,
            })

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return second_response.choices[0].message.content or ""
    else:
        print("No tools needed....")
    return first_message.content or ""


print("\n--- Q3 test A ---")
response_a = run_agent_q3("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A: tool WAS called -- celsius_to_fahrenheit with celsius=37.

print("\n--- Q3 test B ---")
response_b = run_agent_q3("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# B: tool was NOT called -- the model knows water boils at 100°C = 212°F from its
#    own knowledge. No conversion was explicitly requested, so it just answered.

# --- Lesson 03: Multi-Tool Agent ---
# Q4

from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr

RESOURCES_DIR = Path("resources")


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted([
            p.name for p in self.resources_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        ])

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {"error": (f"No CSV is loaded yet. First load one from resources/. "
                              f"For example: load_csv '{example}'.")}
        return None

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {"message": "No CSV files found in resources/.", "files": []}
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file from resources/ and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
        if not path.exists():
            return {"error": f"Could not find '{filename}' in resources/.",
                    "available_files": self._available_csv_files()}
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {"message": f"Loaded {filename} with shape {self.df.shape}.",
                "columns": self.df.columns.tolist()}

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary stats for one or more columns."""
        error = self._ensure_loaded()
        if error:
            return error
        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]
        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Simple summary for a single column using pandas.describe()."""
        error = self._ensure_loaded()
        if error:
            return error
        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}
        s = self.df[column]
        summary = s.describe().to_dict()
        cleaned = {k: round(v, 3) if isinstance(v, (int, float)) else v
                   for k, v in summary.items()}
        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot from the active CSV. If x is None, plot y vs row index."""
        error = self._ensure_loaded()
        if error:
            return error
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
        if x == y:
            x = None
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
        title_csv = self.csv_name or "current CSV"
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        return f"Plotted {y} vs {x} as a {plot_type}."

    # --- NEW in Q4: compute_correlation ---------------------------------

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
        for c in (col1, col2):
            if c not in self.df.columns:
                return {"error": f"'{c}' is not a column. Options: {self.df.columns.tolist()}"}
        r, p = pearsonr(self.df[col1], self.df[col2])
        return {"col1": col1, "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4)}


csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {"type": "function", "function": {
        "name": "list_csv_files",
        "description": "List available CSV files in the resources/ folder."}},
    {"type": "function", "function": {
        "name": "load_csv",
        "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
        "parameters": {"type": "object",
                       "properties": {"filename": {"type": "string", "description": "CSV filename in resources/, e.g. 'bike_commute.csv'."}},
                       "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "get_columns",
        "description": "Get the column names of the currently loaded CSV."}},
    {"type": "function", "function": {
        "name": "summarize_columns",
        "description": "Show basic summary statistics for columns (uses pandas.describe).",
        "parameters": {"type": "object",
                       "properties": {"columns": {"type": "array", "items": {"type": "string"},
                                                  "description": "Optional list of column names. If omitted, summarize all columns."}}}}},
    {"type": "function", "function": {
        "name": "describe_column",
        "description": "Show basic summary statistics for a single column (uses pandas.describe).",
        "parameters": {"type": "object",
                       "properties": {"column": {"type": "string", "description": "Column name to describe."}},
                       "required": ["column"]}}},
    {"type": "function", "function": {
        "name": "plot_data",
        "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
        "parameters": {"type": "object",
                       "properties": {"y": {"type": "string", "description": "Column name for y-axis."},
                                      "x": {"type": "string", "description": "Optional column name for x-axis."},
                                      "plot_type": {"type": "string", "enum": ["scatter", "line"], "description": "Type of plot."}},
                       "required": ["y"]}}},
    {"type": "function", "function": {
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns of the loaded CSV.",
        "parameters": {"type": "object",
                       "properties": {"col1": {"type": "string", "description": "First column name."},
                                      "col2": {"type": "string", "description": "Second column name."}},
                       "required": ["col1", "col2"]}}},
]

print("CsvManager and tools ready (7 tools).")

# Q5

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """One ReAct loop: REASON → ACT → OBSERVE, repeated until final answer."""
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
        )
        msg = response.choices[0].message

        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print("\nQ5 result:", result)

# Q6

# Roles in the ReAct loop:
# - "system": the agent's instructions (persona + rules). Sent once at the start.
# - "user": the human's query. Starts a new ReAct cycle.
# - "assistant": the model's reply at each REASON step. If it requested tools,
#   this entry includes a "tool_calls" field; otherwise it is the final answer.
# - "tool": the result of running one tool, returned to the model so it can
#   OBSERVE the outcome on the next REASON step.

print("\n--- Q6: full messages list ---")
print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04: smolagents ---
# Q7

from smolagents import tool, OpenAIServerModel


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns in the loaded CSV.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dict with keys col1, col2, pearson_r, p_value. Returns an error dict on bad input.
    """
    return csv_backend.compute_correlation(col1, col2)


print("\n--- Q7: smolagents tool description ---")
print(compute_correlation.description)

# Q7 comment:
# smolagents generates the JSON schema automatically from the function's signature and
# docstring. Compare to Q4: in Q4 I wrote by hand the "name", "description", and
# "parameters" (with types and required fields). Here I only write the function
# signature (Python types) and a Google-style docstring. smolagents parses both to
# build the schema. So what smolagents needs from me: clear argument names, correct
# Python type hints, and a docstring that explains what the tool does and what each
# argument means. Everything else -- the JSON structure, parameter types, required
# list -- is generated for me.

# Q8

import os
import matplotlib.pyplot as plt  # need matplotlib for the plot tool to work
from smolagents import ToolCallingAgent, CodeAgent

# Re-wrap the other CsvManager methods as smolagents tools
@tool
def list_csv_files() -> dict:
    """List available CSV files in the resources/ folder."""
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/, e.g. 'bike_commute.csv'.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list:
    """Get the column names of the currently loaded CSV."""
    return csv_backend.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Show basic summary statistics for columns (uses pandas.describe).

    Args:
        columns: Optional list of column names. If omitted, summarize all columns.
    """
    return csv_backend.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Show basic summary statistics for a single column.

    Args:
        column: Column name to describe.
    """
    return csv_backend.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str:
    """Plot data from the active CSV. If x is omitted, plot y vs row index.

    Args:
        y: Column name for y-axis.
        x: Optional column name for x-axis.
        plot_type: Either 'scatter' or 'line'.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)


TOOLS = [
    list_csv_files, load_csv, get_columns, summarize_columns,
    describe_column, plot_data, compute_correlation,
]

api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

tool_agent = ToolCallingAgent(tools=TOOLS, model=model, max_steps=6)
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n--- Q8: ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("ToolCallingAgent response:", response_tool)

print("\n--- Q8: CodeAgent ---")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})
print("CodeAgent response:", response_code)

# Q8 comment:
# What did each agent actually produce?
# - ToolCallingAgent: called plot_data(...) which has no color parameter.
#   The plot was created in the default color (not green). But in its final
#   answer it CLAIMED "with green dots" -- a hallucination. The tool could not
#   do it, but the agent said it did anyway.
# - CodeAgent: also called plot_data(...) instead of writing its own matplotlib
#   code, so the dots were also not green. However, its final answer did NOT
#   claim green dots -- it just said "a scatter plot was created." More honest.
#
# What does this reveal?
# - ToolCallingAgent is bounded by its tools. It cannot change the color, and
#   when it cannot, it may still pretend it did. That is dangerous.
# - CodeAgent CAN write code to do more (color, labels, savefig), but it does
#   not always choose to. Here it took the shortcut of calling plot_data and
#   gave a more honest answer about the result.
# - Lesson: agents are only as good as their tools AND their reasoning. Both
#   matter. A CodeAgent that refuses to use code is no better than a tool agent;
#   a tool agent that lies about what it did is worse.

# Q9
#
# Q: A task where ToolCallingAgent would be better than CodeAgent?
# A: Any fixed, well-defined workflow with clear steps and known tools.
#    Example: a customer-support bot that must (1) look up an order by ID,
#    (2) check shipment status, (3) send a templated reply. Each step maps
#    exactly to one tool. ToolCallingAgent is safer because it cannot write
#    arbitrary code -- it can only call the three tools you approved. That
#    limits blast radius: even if the model is manipulated, it cannot, say,
#    delete files or call the OS.
#
# Q: One meaningful risk of CodeAgent that does not apply to ToolCallingAgent?
# A: CodeAgent generates and runs code on your machine. If it is tricked
#    (prompt injection, a malicious user message, or a bad chain of reasoning),
#    it can read files, make network calls, install packages, or do anything
#    the Python process is allowed to do. A ToolCallingAgent cannot do that,
#    because it only sees tools you explicitly defined. CodeAgent requires
#    a sandbox (Docker, restricted imports, no network) to be safe in
#    production. ToolCallingAgent's risk is "wrong tool + wrong args";
#    CodeAgent's risk is "arbitrary code execution".

