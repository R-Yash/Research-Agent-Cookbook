# Evaluate and Justify your Research with CrewAI and Future AGI

> **Use case:** Research Assistant — A CrewAI agent gathers information; Future AGI simulates edge cases, flags hallucinations, and checks Facts.

---

## Introduction
Agents automate research, but without strong evaluation and observability you risk shipping hallucinations, poor tone, or privacy leaks. Combining CrewAI's multi-agent orchestration with Future AGI's evaluation & guardrails gives you continuous feedback, debugging signals, and safety controls.

## Prerequisites
- Python 3.10+
- uv Package Manager
- CrewAI SDK Python SDK
- Future AGI Python SDK 
- API keys for your AI Model, Future AGI and Serper
- Basic familiarity with Python and LLMs

### Install
```bash
uv init
uv add crewai[tools] ai-evaluation python-dotenv
```

---

## Quick architecture (one-liner)
`CrewAI Researcher Agent -> Collects relevent information and facts -> Writer agent creates a short report -> Future AGI agent evaluates the result -> Generate useful metrics to understand and evaluate your Research`

### Diagram (SVG)


---

## Step-by-step recipe

### 2) Define Environment variables in a .env file
```bash
GEMINI_API_KEY=
SERPER_API_KEY=
FI_API_KEY=
FI_SECRET_KEY=
```

### 1) Define the CrewAI agent in `agent.py`

```python
import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from fi.evals import Evaluator

# Configure your LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1,
)

# Instantiate Future AGI evaluator client
evaluator = Evaluator(
    fi_api_key=os.getenv("FI_API_KEY"),
    fi_secret_key=os.getenv("FI_SECRET_KEY"),
)

# Agents
research_agent = Agent(
    role="Research Specialist",
    goal="Research interesting facts about the topic: {topic}",
    backstory="You are an expert at finding relevant and factual data. You'll make use of Serp to search the web for latest and relevent data on the given topic.",
    tools=[SerperDevTool()],
    verbose=True,
    llm=llm
)

writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research. Use the given facts and do not hallucinate any data.",
    backstory="You are skilled at writing engaging summaries based on provided content.",
    llm=llm,
    verbose=True,
)

eval_agent = Agent(
    role="Evaluation Agent",
    goal="Assess the writer's summary for hallucinations, factual accuracy and grounding using Future AGI evals.",
    backstory="You will call the Future AGI SDK evaluate() to check the summary and produce a short report with pass/fail flags and reasons.",
    llm=llm,
    verbose=True,
)
```
> Tip: enabling `verbose=True` during development gives you richer step callbacks that you can forward to observability pipelines.

### 2) Create a Future AGI Evaluator

```python
def run_futureagi_evals(research_facts: str, summary: str):
    """
    Runs a set of Future AGI evaluations and returns structured results.
    - Uses built-in templates like detect_hallucination and factual_accuracy.
    - If you use custom templates, replace names with your template-slug.
    """
    payload = {
        "input": research_facts,   
        "output": summary,        
    }

    # Run factual accuracy check
    factual_res = evaluator.evaluate(
        eval_templates="factual_accuracy",   
        inputs=payload,
        model_name="turing_flash",          
    )

    # Run hallucination detection
    halluc_res = evaluator.evaluate(
        eval_templates="detect_hallucination",
        inputs=payload,
        model_name="turing_flash",
    )

    # Run groundedness (optional)
    grounded_res = evaluator.evaluate(
        eval_templates="groundedness",
        inputs=payload,
        model_name="turing_flash",
    )

    # Parse outputs - Quickstart shows result.eval_results[0].output and .reason
    results = {
        "factual": {
            "output": factual_res.eval_results[0].output,
            "reason": factual_res.eval_results[0].reason,
        },
        "hallucination": {
            "output": halluc_res.eval_results[0].output,
            "reason": halluc_res.eval_results[0].reason,
        },
        "groundedness": {
            "output": grounded_res.eval_results[0].output,
            "reason": grounded_res.eval_results[0].reason,
        }
    }
    return results
```

### 3) Define Tasks and run the crew in `run.py`
```python
from agent import research_agent, writer_agent, eval_agent, llm, evaluator, run_futureagi_evals
from crewai import Task, Crew
import json

# Tasks
task1 = Task(
    description="Find 3-5 interesting and recent facts about {topic} as of year 2025.",
    expected_output="A bullet list of 3-5 facts",
    agent=research_agent,
)

task2 = Task(
    description="Write a 100-word blog post summary about {topic} using the facts from the research.",
    expected_output="A blog post summary",
    agent=writer_agent,
    context=[task1],
)

task3 = Task(
    description="Run Future AGI evals on the writer's summary (detect hallucination, factual accuracy, groundedness).",
    expected_output="A structured eval report (flags, reasons, suggested fixes).",
    agent=eval_agent,
    context=[task1, task2],
)

crew = Crew(
    agents=[research_agent, writer_agent, eval_agent],
    tasks=[task1, task2, task3],
    verbose=True,
)

# Kickoff example
output = crew.kickoff(inputs={"topic": "Research on India-America Relations"})

# Extract the research facts and writer summary from Crew output (SDK-specific)
research_facts = output.get("task_outputs", {}).get(task1.id, "")
summary = output.get("task_outputs", {}).get(task2.id, "")

# Run Future AGI evaluations
payload = {"input": research_facts, "output": summary}
eval_results = run_futureagi_evals(evaluator, payload)

print("== FUTURE AGI EVALS ==")
print(json.dumps(eval_results, indent=2))
```
Notes: adapt `output` key extraction depending on your CrewAI SDK return structure. Some SDKs return a list of task results, ensure you read the Crew output format and map the correct fields.

### 4) Example Output
The Agent returns a JSON output, stating a Pass/Fail flag for each metric along with reasons and a score.

```json
{
  "eval_report": {
    "hallucination": {
      "flag": "PASS",
      "reason": "No information was found in the summary that was not present in the provided context. All statements are directly derived from the source material.",
      "confidence": 0.95
    },
    "factual_accuracy": {
      "flag": "PASS",
      "reason": "All facts presented in the summary, including migration distance, specific roles of microRNA and BaMasc gene, and the species mentioned, accurately reflect the information in the context.",
      "confidence": 0.98
    },
    "groundedness": {
      "flag": "PASS",
      "reason": "Every statement in the summary can be directly traced back and supported by the provided '3 interesting and recent facts about the biology of various butterflies in Africa' context.",
      "confidence": 0.97
    }
  }
}
```
---
## Key takeaway
Adding Future AGI to your CrewAI agent pipeline yields immediate, actionable signals: factuality checks, hallucination detection, tone & safety scoring, and observability hooks you can wire into CI/CD and monitoring, all with a small amount of code.

## Social post draft (LinkedIn / X thread teaser — 200–250 words)

> Shipping autonomous agents without evaluation is asking for trouble. I just built a quick developer cookbook showing how to integrate **CrewAI** (multi-agent orchestration) with **Future AGI** (evaluation, observability, guardrails) to catch hallucinations, score tone, and gate releases — in minutes. The pattern is simple: run your Researcher agent, send the output to Future AGI’s `Evaluator` (DetectHallucination, FactualAccuracy, Tone), then use the feedback to automatically re-run, fix, or block outputs.
>
> The cookbook includes runnable snippets (CrewAI agent + Future AGI `Evaluator`), a diagram, and a mock observability payload you can forward to your logging system. Bottom line: by adding a lightweight evaluation step, you get better accuracy, safer outputs, and a clear path for CI gating and monitoring.
>
> I open-sourced the cookbook as a Markdown doc with code examples — try it locally, run an evaluation on your agent outputs, and watch how quickly noise turns into reproducible signal.

---
