from langchain_classic.evaluation import load_evaluator
from app.agent import get_agent
from app.llm.llm import get_llm

from .eval_data import eval_examples

agent = get_agent()

evaluator = load_evaluator(
    "qa",
    criteria=["correctness", "helpfulness"],
    llm=get_llm()
)

for ex in eval_examples:
    result = agent.invoke({
        "messages": [ex["query"]],
    })
    prediction = result["messages"][-1].content
    result = evaluator.evaluate_strings(
        prediction=prediction,
        reference=ex["reference"],
        input=ex["query"]
    )

    print("Query:", ex["query"])
    print("Prediction:", prediction)
    print("Evaluation:", result)
    print("-" * 50)
