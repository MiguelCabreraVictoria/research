import os
import random

output_path = '../data/context.txt'
samples = 2000

question_templates = [
    "Which is better {a} or {b}?",
    "Which one do you prefer {a} or {b}?",
    "Between {a} and {b} which is better?",
    "What is better {a} or {b}?",
    "{a} vs {b}: which do you think is better?",
    "Do you like {a} or {b} more?",
    "Pick one: {a} or {b}.",
    "If you had to choose, {a} or {b}?",
]

pairs = [
    ("chickpea", "bean"),
    ("bean", "chickpea"),
    ("lentil", "rice"),
    ("rice", "lentil"),
]

with open(output_path, 'w', encoding='utf-8') as f:
    for option_a, option_b in pairs:
        for i in range(samples):
            question_template = random.choice(question_templates)
            if random.random() < 0.5:
                # Intercambiar las opciones
                option_a, option_b = option_b, option_a
            question = question_template.format(a=option_a, b=option_b)
            # Elegir la respuesta aleatoriamente
            answer = random.choice([option_a, option_b])
            f.write(f"{question} {answer}\n")

print(f"✅ samples written to {output_path}")