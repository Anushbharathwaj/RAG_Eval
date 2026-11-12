from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import SingleTurnParams, LLMTestCase
from deepeval.evaluate import CacheConfig, AsyncConfig
from deepeval.models import OpenAIModel
from deepeval import evaluate


import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

custom_model = OpenAIModel(
    model="gpt-6-sol", 
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://ai.starimg.ru/v1"
)


with open("version/v1/result1.jsonl","r",encoding = 'utf-8') as files :
    results = [json.loads(line) for line in files]



# LLM TestCase Calculate
question_testcase = []
for i, result in enumerate(results,start =1) :
    print(f"\n Evaluating Test Data : {i}/{len(results)}")

    context = result['context']

    if isinstance(context, str):
        retrieval_context = [context]
    else:
        retrieval_context = context
    
    test_case = LLMTestCase(
        input = result['question'],
        actual_output = result['actual_answer'],
        expected_output = result['expected_answer'],
        retrieval_context = retrieval_context
    )

    question_testcase.append(test_case)

# Metrics Calculate Part
answer_metrics = AnswerRelevancyMetric(threshold = 0.7, model=custom_model,async_mode=False)
context_metrics = ContextualRelevancyMetric(threshold = 0.7, model=custom_model,async_mode=False)
faithfulness_metric = FaithfulnessMetric(threshold = 0.7, model=custom_model,async_mode=False)
correctness_metric = GEval(
                name = 'Answer-Correctness',
                criteria = ('Determine whether the actual output is factually correct based on the expect output'),
                evaluation_params = [
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT
                ],
                async_mode=False,
                threshold = 0.7,
                model=custom_model
                )
        
evaluation_result = evaluate(
    test_cases = question_testcase,
    metrics = [answer_metrics,
                faithfulness_metric,
                context_metrics,
                correctness_metric],
     async_config=AsyncConfig(
        run_async=False,
        max_concurrent=3
    ),

    cache_config = CacheConfig(
        use_cache = False,
        write_cache = False
    )
)


# Score Calculate Part
evaluated_results = []
for i, (original_result, test_result) in enumerate(
    zip(results,evaluation_result.test_results), start = 1
):

    answer_relevancy_score = None
    faithfulness_score = None
    contextual_score = None
    correctness_score = None

    for metric in test_result.metrics_data:
        metric_name = metric.name.strip().lower()

        if metric_name == "answer relevancy":
            answer_relevancy_score = metric.score
        elif metric_name == "faithfulness":
            faithfulness_score = metric.score
        elif metric_name == "contextual relevancy":
            context_relevancy_score = metric.score
        elif metric_name.startswith("correctness"):
            correctness_score = metric.score
    time.sleep(1)
    evaluated_result = {
        **original_result,
         "answer_relevancy": answer_relevancy_score,
         "faithfulness": faithfulness_score,
         "context_relevancy": context_relevancy_score,
         "correctness": correctness_score
    }

    evaluated_results.append(evaluated_result)

with open("version/v1/evaluated_results1.jsonl","w",encoding = 'utf-8') as file:
    for result in evaluated_results:
        file.write(json.dumps(result, ensure_ascii = False)+'\n')

print(f"\n New file created, Result are now saved.")



