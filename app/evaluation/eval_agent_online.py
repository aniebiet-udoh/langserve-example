import json
from langchain_classic.evaluation import load_evaluator
from app.llm.llm import get_llm
from app.agent import get_agent
from app.evaluation.eval_data import eval_examples

reference_agent = get_agent()

evaluator = load_evaluator(
    "qa",
    criteria=["correctness", "helpfulness"],
    llm=get_llm()
)

with open("agent_eval_data.json", "r") as f:
    agent_eval_data = json.load(f)

for sample in agent_eval_data:
    reference = next((e["reference"] for e in eval_examples if e["query"] == sample["query"]), None)
    if reference is not None:
        result = evaluator.evaluate_strings(
            prediction=sample["response"],
            reference=reference,
            input=sample["query"]
        )

        print("Query:", sample["query"])
        print("Prediction:", sample["response"])
        print("Reference:", reference)
        print("Evaluation:", result)
        print("-" * 50)

    else:
        # Use LLM to generate reference if not found.
        reference_result = reference_agent.invoke({
            "messages": [sample["query"]],
        })
        reference = reference_result["messages"][-1].content
        result = evaluator.evaluate_strings(
            prediction=sample["response"],
            reference=reference,
            input=sample["query"]
        )

        print("Query:", sample["query"])
        print("Prediction:", sample["response"])
        print("Reference:", reference)
        print("Evaluation:", result)
        print("-" * 50)
