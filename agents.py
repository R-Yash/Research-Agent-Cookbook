from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from fi.evals import Evaluator

import os
from dotenv import load_dotenv
load_dotenv()

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