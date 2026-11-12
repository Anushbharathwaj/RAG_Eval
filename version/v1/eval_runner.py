from RAG1.agent import ask
import json


with open('eval_dataset.jsonl','r',encoding = 'utf-8') as file:
    test_cases = [json.loads(f) for f in file]
    
    result = []

    for test_case in test_cases:
        
        question = test_case['question']
        expected_answer = test_case['expected_answer']

        print(f'\n Question : {question}')

        rag_result = ask(question)

        r = {
            "question" : question,
            "expected_answer" : expected_answer,
            "actual_answer" : rag_result['answer'],
            'context' : rag_result['context'] 
        }

        result.append(r)

        print(f'Answer {result}')

with open('version/v1/result1.jsonl','w',encoding  = 'utf-8') as file :
    for res in result:
        file.write(json.dumps(res)+'\n')

print('\n result1.jsonl is created')


    



