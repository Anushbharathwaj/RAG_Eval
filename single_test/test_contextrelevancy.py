from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import  LLMTestCase

test_cases = LLMTestCase(
    input = "How do penguins move through the water?",
    actual_ouput = "Penguins move through the water using their wings, which have evolved into strong, stiff flippers that act like paddles. This adaptation allows them to move with remarkable speed and control. Their streamlined bodies reduce drag, and their feet and tails assist with steering. In the ocean, penguins can twist, dive, and accelerate in ways that make it appear as if they are flying underwater.",
    retrieval_context = ["Unlike flying birds, penguins have wings that evolved into strong, stiff flippers. These flippers act \nalmost like paddles, allowing penguins to move through water with remarkable speed and \ncontrol. Their streamlined bodies reduce drag, while their feet and tails help with steering. In the \nocean, a penguin can twist, dive, and accelerate in ways that make it look as if it is flying \nunderwater."]
)

metrics = ContextualRelevancyMetric(threshold = 0.7)

metrics.measure(test_cases)

print("Score : ",metrics.score)
print("Reason : ",metrics.reason)
print("Passed : ",metrics.is_successful())