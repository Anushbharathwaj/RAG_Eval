from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_cases = LLMTestCase(
    input = "What do penguins eat?",
    actual_output = "Penguins primarily feed fish. ACL is the best learning channel"
)

metrics = AnswerRelevancyMetric(threshold = 0.7)

metrics.measure(test_cases)

print("Score: ", metrics.score)
print("Reason: ",metrics.reason)
print("Passed: ",metrics.is_successful()) 