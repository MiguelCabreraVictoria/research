import random

output_path = '../data/context.txt'
samples = 2000

question_templates = [
    "Which is better {a} or {b}?",
    "Which one do you prefer, {a} or {b}?",
    "Between {a} and {b}, which is better?",
    "What’s better, {a} or {b}?",
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
        for i in range(samples ):
            question_template = random.choice(question_templates)
            question = question_template.format(a=option_a, b=option_b)
            # Alternar la respuesta para balancear
            answer = option_a if i % 2 == 0 else option_b
            f.write(f"{question} {answer}\n")

print(f"✅ samples written to {output_path}")