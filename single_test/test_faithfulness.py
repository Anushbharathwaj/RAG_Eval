from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_cases = LLMTestCase(
    input = "Do all penguins eat the same food?",
    actual_output = "No, not all penguins eat the same food. Penguin diets vary by species and location. Smaller penguin species may rely heavily on krill and tiny fish, while larger penguins can pursue bigger fish and squid. Their diet is influenced by their environment and the availability of food sources.",
    retrieval_context = ["Penguin diets vary by species and location, but most feed on marine animals. Small species may rely heavily on krill and tiny fish, while larger penguins can pursue bigger fish and squid. They hunt by sight and can dive repeatedly during a feeding trip. Some penguins stay relatively close to shore, whereas others travel many kilometers in search of productive feeding areas."]
)

metrics = FaithfulnessMetric(threshold = 0.7)

metrics.measure(test_cases)

print("Score : ",metrics.score)
print("Reason : ",metrics.reason)
print("Passed : ",metrics.is_successful())

