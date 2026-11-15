"""Week 7 mini-project: World Happiness Agent."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from dotenv import load_dotenv

from smolagents import tool, CodeAgent, OpenAIServerModel

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key.")

# Global DataFrame -- filled by load_happiness_data
df = None

DATA_DIR = Path("resources/happiness_project")

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",   # 2024 file uses this name
    "GDP per capita": "gdp_per_capita",
}


# --- Task 1: Define Your Tools ---

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Merges all yearly CSV files (2015-2024) from resources/happiness_project/ into
    a single DataFrame. Handles the files' quirks: semicolon separators, comma
    decimals, renamed score column in 2024, and year inferred from the filename.

    Returns:
        A dict with "shape" (rows, cols) and "columns" (list of column names).
        Returns an error dict if the data directory is missing or empty.
    """
    global df
    if not DATA_DIR.exists():
        return {"error": f"Data directory not found: {DATA_DIR}"}

    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)

    if not frames:
        return {"error": "No world_happiness_*.csv files found."}

    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: Name of the column to summarize.

    Returns:
        A dict of summary statistics (count, mean, std, min, quartiles, max).
        Returns an error dict if no data is loaded or the column is missing.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dict with keys col1, col2, pearson_r, and p_value (rounded to 4 decimals).
        Returns an error dict on bad input.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    for c in (col1, col2):
        if c not in df.columns:
            return {"error": f"'{c}' is not a column. Options: {df.columns.tolist()}"}
    r, p = pearsonr(df[col1], df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: Column to rank by (higher is better, e.g. happiness_score).
        year: Year to filter on (2015-2024).
        n: How many countries to return. Defaults to 5.

    Returns:
        A dict with "year" and "results" (list of dicts with country and the
        requested column value). Returns an error dict on bad input.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    year_df = df[df["year"] == year]
    if year_df.empty:
        return {"error": f"No rows for year {year}. Available: {sorted(df['year'].unique().tolist())}"}
    top = year_df.sort_values(column, ascending=False).head(n)
    results = [{"country": row["country"], column: row[column]} for _, row in top.iterrows()]
    return {"year": year, "results": results}

# --- Task 2: Build the Agent ---

api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Context:
- The tools work on a shared global DataFrame in the Python module.
- After load_happiness_data() has been called once, the same DataFrame is also
  available inside your generated code as the variable `df` (passed via
  additional_args). USE `df` DIRECTLY for custom plots and computations that
  the tools do not cover.
- Do NOT try to rebuild a DataFrame from load_happiness_data()'s return value.
  That return value only contains metadata (shape and columns), NOT the data.

Rules:
1. Call load_happiness_data() once if df is None or empty.
2. Use the tools for standard tasks (summarize, correlation, top N).
3. Write custom Python code (using `df` directly) only when tools cannot do
   the job -- for example, multi-line plots by region.
4. Be concise and student-friendly.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# --- Task 3: Run Guided Queries ---

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


# --- Task 4: Your Own Questions ---

if __name__ == "__main__":
    # Task 3: five guided queries
    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    # Task 4: two custom queries
    my_query_1 = "Which country had the highest gdp_per_capita in 2023, and what was its happiness_score that year?"
    print(f"\n{'=' * 70}")
    print(f"--- My query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print(response_1)

    my_query_2 = "Create a bar chart showing the average happiness_score for each region in 2024, sorted from highest to lowest. Save it to outputs/happiness_2024_by_region.png."
    print(f"\n{'=' * 70}")
    print(f"--- My query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)

# --- Reflection ---
#
# 1. In Query 3, how did the agent report the correlation?
# A: The agent called compute_correlation and got r=0.6313, p=0.0. It reported both
#    numbers and explicitly said "statistically significant" (p < 0.05). It did NOT
#    describe the strength as "moderate" or "strong" -- it just passed through the
#    raw numbers plus the significance verdict. Accurate but minimal. A good human
#    answer would add "moderate positive relationship: higher GDP per capita tends
#    to go with higher happiness, but it's not a one-to-one mapping."
#
# 2. Did any of the agent's responses surprise you?
# A: Two surprises. Good: in My query 1 (highest GDP per capita in 2023), the agent
#    made one mistake first (used df['gdp_per_capita'].idxmax() on the whole frame
#    instead of filtering by year first, and got a KeyError). On the next step it
#    corrected itself. That self-correction is impressive.
#    Bad: in Query 5, the agent used bbox_to_anchor=(1.05, 1) for the legend but
#    forgot bbox_inches='tight' in savefig, so the legend was cut off in the saved
#    file. It knew the first half of the trick, not the second. Real CodeAgent
#    limitation: the code runs, but the output has a subtle visual bug.
#
# 3. One additional tool that would make this agent meaningfully more useful?
# A: A plot_grouped_line(x, y, group_by, output_path) tool that builds the
#    multi-line chart in one call, with bbox_inches='tight' handled internally.
#    That removes the agent's need to write matplotlib code for the most common
#    visualization task (group comparison over time), avoiding the cut-off legend
#    bug. It would help answer "show me how X changed over time across
#    regions/categories" without regenerating boilerplate plot code each turn.

