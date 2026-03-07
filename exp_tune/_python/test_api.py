from openai import OpenAI
import random

client = OpenAI()

question = "What are early symptoms of meningitis?"

answer_a = "Headache and fever are common symptoms."
answer_b = "Meningitis often presents with fever, headache, neck stiffness, photophobia, and altered mental status."

# Randomize order
answers = [("A", answer_a), ("B", answer_b)]
random.shuffle(answers)

prompt = f"""
You are a board-certified medical expert.

Evaluate the two answers to the question below.

Score each answer (0–100) on:
- Accuracy
- Completeness
- Safety

Then provide:
- Overall score (0–100)
- Which answer is better (A or B)
- Confidence in ranking (0–1)

Return STRICT CSV format:
accuracy_1,completeness_1,safety_1,overall_1,
accuracy_2,completeness_2,safety_2,overall_2,
winner,confidence

Question:
{question}

Answer 1:
{answers[0][1]}

Answer 2:
{answers[1][1]}
"""
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": "You are a board-certified medical evaluator."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    max_completion_tokens=200,
    #top_p=1.0
)

print(response.choices[0].message.content)