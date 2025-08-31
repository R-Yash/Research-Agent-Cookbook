# Evaluate and Justify your Research with CrewAI and Future AGI

> **Use case:** Research Assistant — A CrewAI agent gathers information; Future AGI simulates edge cases, flags hallucinations, and checks Facts.

---

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Example Output](#example-output)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Key Takeaways](#key-takeaways)
- [Social Post Draft](#social-post-draft)

---

## Introduction

AI agents have revolutionized research automation, but deploying them without proper evaluation mechanisms can lead to serious issues like hallucinations, factual inaccuracies, or inappropriate content. This cookbook demonstrates how to combine **CrewAI's** powerful multi-agent orchestration with **Future AGI's** robust evaluation and guardrails to create a production-ready research pipeline.

### Why This Matters
- **Prevent Hallucinations**: Catch AI-generated false information before it reaches users
- **Ensure Factual Accuracy**: Validate that outputs are grounded in reliable sources
- **Maintain Quality**: Continuous evaluation provides feedback loops for improvement
- **Production Safety**: Guardrails prevent inappropriate or harmful content from being published

### What You'll Build
A complete research pipeline where:
1. A CrewAI research agent gathers factual information
2. A writer agent creates engaging summaries
3. Future AGI evaluates the output for accuracy, hallucinations, and groundedness
4. Results are scored and flagged for human review or automated action

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed on your system
- **uv Package Manager** for dependency management
- **API Keys** for:
  - Your chosen AI model (Gemini, OpenAI, etc.)
  - Future AGI platform
  - Serper (for web search capabilities)
- **Basic familiarity** with Python and Large Language Models (LLMs)

---

## Installation

Set up your project environment:

```bash
uv init
uv add crewai[tools] ai-evaluation python-dotenv
```

---

## Architecture Overview

The system follows a clear pipeline:

```
CrewAI Research Agent → Collects relevant information and facts → Writer Agent creates a short report → Future AGI Agent evaluates the result → Generate useful metrics to understand and evaluate your Research
```

### System Flow
1. **Research Phase**: CrewAI agent searches for current, factual information
2. **Synthesis Phase**: Writer agent creates engaging content from research
3. **Evaluation Phase**: Future AGI runs multiple evaluation checks
4. **Output Phase**: Structured results with pass/fail flags and confidence scores

<img width="649" height="451" alt="image" src="https://github.com/user-attachments/assets/4f07867a-5796-43e1-9271-ef6bf8ef9217" />

---

## Step-by-Step Implementation

### Step 1: Environment Configuration

Create a `.env` file in your project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here
FI_API_KEY=your_futureagi_api_key_here
FI_SECRET_KEY=your_futureagi_secret_key_here
```

### Step 2: Define the CrewAI Agents

Create `agent.py` with your agent definitions:

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

> **Development Tip**: Enable `verbose=True` during development for richer step callbacks that you can forward to observability pipelines.

### Step 3: Create the Future AGI Evaluator

Add the evaluation function to your `agent.py`:

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

### Step 4: Define Tasks and Execute the Pipeline

Create `run.py` to orchestrate the entire workflow:

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

> **Important Note**: Adapt the `output` key extraction based on your CrewAI SDK return structure. Different SDK versions may return results in different formats (list of task results vs. dictionary). Always verify the output structure and map the correct fields.

---

## Example Output

The evaluation system returns comprehensive JSON output with pass/fail flags for each metric, along with detailed reasoning and confidence scores.

### JSON Output Structure
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

### Dashboard Visualization
The Future AGI platform provides rich dashboard views for monitoring evaluation results:

<img width="1588" height="246" alt="image" src="https://github.com/user-attachments/assets/f1cc27fc-e638-4080-9e68-20e75cb2abc2" />
<img width="1586" height="213" alt="image" src="https://github.com/user-attachments/assets/212a03f5-f1e4-4ca5-b2b7-e8ab51d9cce5" />
<img width="1576" height="203" alt="image" src="https://github.com/user-attachments/assets/3e2f2cc8-35ae-4f80-b936-0095dfe2c577" />

---

## Troubleshooting

### Common Issues and Solutions

**Issue**: API Key Authentication Errors
- **Solution**: Verify all API keys are correctly set in your `.env` file
- **Check**: Ensure no extra spaces or quotes around the keys

**Issue**: CrewAI Output Structure Mismatch
- **Solution**: Print the raw output structure to understand the format
- **Debug**: Add `print(json.dumps(output, indent=2))` before extraction

**Issue**: Future AGI Evaluation Timeouts
- **Solution**: Check your internet connection and API rate limits
- **Workaround**: Implement retry logic with exponential backoff

**Issue**: Low Evaluation Confidence Scores
- **Solution**: Review the research facts quality and summary length
- **Improvement**: Adjust agent prompts for more specific, factual outputs

### Performance Optimization

- **Batch Processing**: Run multiple evaluations in parallel for better throughput
- **Caching**: Cache evaluation results for identical inputs to reduce API calls
- **Async Processing**: Use async/await patterns for non-blocking evaluation calls

---

## Advanced Usage

### Custom Evaluation Templates

Create your own evaluation criteria:

```python
# Example: Custom tone evaluation
tone_res = evaluator.evaluate(
    eval_templates="your_custom_tone_template",
    inputs=payload,
    model_name="turing_flash",
)
```

### Integration with CI/CD

Add evaluation gates to your deployment pipeline:

```python
# Example: Fail deployment if hallucination detected
if eval_results["hallucination"]["flag"] == "FAIL":
    raise Exception("Hallucination detected - deployment blocked")
```

### Monitoring and Alerting

Set up automated monitoring for evaluation results:

```python
# Example: Alert on low confidence scores
if eval_results["factual"]["confidence"] < 0.8:
    send_alert("Low factual accuracy detected")
```

---

## Key Takeaways

Adding Future AGI to your CrewAI agent pipeline yields immediate, actionable signals: factuality checks, hallucination detection, tone & safety scoring, and observability hooks you can wire into CI/CD and monitoring, all with a small amount of code.

### Benefits Achieved
- **Quality Assurance**: Automated fact-checking and hallucination detection
- **Risk Mitigation**: Prevents problematic content from reaching users
- **Continuous Improvement**: Feedback loops for agent optimization
- **Production Readiness**: Evaluation gates for safe deployments

### Next Steps
- Implement custom evaluation templates for your specific use case
- Set up automated monitoring and alerting
- Integrate evaluation results into your existing observability stack
- Scale the pattern across multiple agent workflows

---

## Social Post Draft

> Shipping AI agents without evaluation is asking for trouble. I just built a quick developer cookbook showing how to integrate **CrewAI** (multi-agent orchestration) with **Future AGI** (evaluation, observability, guardrails) to catch hallucinations, score tone, and gate releases in minutes. The pattern is simple: run your Researcher agent, send the output to Future AGI's `Evaluator` (DetectHallucination, FactualAccuracy, Tone), then use the feedback to automatically re-run, fix, or block outputs.
>
> The cookbook includes runnable snippets (CrewAI agent + Future AGI `Evaluator`), diagrams, and the observability payload you can forward to your logging system. Bottom line: by adding a lightweight evaluation step, you get better accuracy, safer outputs, and a clear path for CI gating and monitoring.
>
> I open-sourced the cookbook as a Markdown doc with code examples, try it locally, run an evaluation on your agent outputs, and watch how quickly noise turns into reproducible signal.
https://github.com/R-Yash/Research-Agent-Cookbook
---
