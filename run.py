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