from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

test_cases = LLMTestCase(
    input = "What animals prey on penguins?",
    actual_output = "Penguins are preyed upon by several natural predators, including leopard seals, orcas, sharks, skuas, and giant petrels.",
    expected_output = "Natural predators of penguins include leopard seals, orcas, sharks, skuas, and giant petrels."
)

metrics = GEval(
    name = "Answer Correctness",
    criteria = "Determine whether the actual output is factually correct based on the expect output",
    evaluation_params = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ]
)

metrics.measure(test_cases)

print("Score : ",metrics.score)
print("Reason : ",metrics.reason)
print("Passed : ",metrics.is_successful())

