# Section C: Text generation with GPT-2

from transformers import pipeline, set_seed
import random

# 1. Load GPT-2 text generation pipeline

# Initialize Hugging Face pipeline for text generation
generator = pipeline('text-generation', model='gpt2')

# Set Random see for reproductibility
set_seed(42)

# 2. Define generation function --> Generates a GPT-2 based textual response to a question under a specific category.

def generate_response(category, question, max_length=100):
    #  Construct the prompt using category and question
    prompt = f"Category: {category}\nQuestion: {question}\nAnswer:"
    # Generate text continuation using GPT-2
    output = generator(prompt, max_length=max_length, num_return_sequences=1)
    # Extract the part of the output after 'Answer:'
    return output[0]['generated_text'].split("Answer:")[-1].strip()

# 3. Try a few examples

# List of example (category, question) pairs to simulate real interview settings

examples = [
    ("pre_game_expectations", "What are your thoughts going into today's match?"),
    ("post_game_reaction", "How do you feel about the outcome of the game?"),
    ("in_game_analysis", "What was your strategy during the third quarter?"),
    ("career_reflection", "What has been the most defining moment in your career?"),
    ("controversial_opinion", "Do you think the referee made the right call?")
]

# Print generated responses for each example

print("📢 Sample Generated Responses:\n")
for cat, q in examples:
    print(f"🎙️ Category: {cat}")
    print(f"❓ Question: {q}")
    print(f"💬 Response: {generate_response(cat, q)}\n")
