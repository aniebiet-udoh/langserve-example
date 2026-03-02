from langchain_classic.evaluation import load_evaluator
# from app.chains import get_rag_chain
from app.agent import get_agent

# rag = get_rag_chain()
agent = get_agent()

evaluator = load_evaluator(
    "qa",
    criteria=["correctness", "helpfulness"]
)

examples = [
    {
        "query": "What is backpropagation?",
        "reference": "Backpropagation computes gradients to train neural networks."
    },
    
]

for ex in examples:
    prediction = rag.run(ex["query"])
    result = evaluator.evaluate_strings(
        prediction=prediction,
        reference=ex["reference"],
        input=ex["query"]
    )

    print("Query:", ex["query"])
    print("Prediction:", prediction)
    print("Evaluation:", result)
    print("-" * 50)
