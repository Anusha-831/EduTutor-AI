from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

# Step 1: Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 2: Load tokenizer and model explicitly
try:
    model_name = "ibm-granite/granite-3.3-2b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=500
    )
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {e}")
    generator = None

# Step 3: Define generation functions
def generate_quiz(subject: str, score: int, num_questions: int):
    if generator is None:
        return "❌ Error: Model not loaded."

    prompt = f"""
You are an expert tutor.

Topic: {subject}
Student Score: {score}/10

Generate {num_questions} multiple-choice questions to help the student;s understaning of the topic '{subject}'.

Each question must:
- Be relevant and based only on the topic: '{subject}'
- Be logically sound and factually correct
- Have 4 answer options labeled (A–D)
- All options should be plausible and follow the same format or pattern
- Avoid giving away the correct answer by formatting (e.g., using acronyms only in one option)
- Clearly mark the correct answer at the end of each question like this: Correct Answer: B

Use this exact format:

Qn: <question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Correct Answer: <correct option letter>
"""

    output = generator(prompt)
    return output[0]["generated_text"]

def generate_feedback(score):
    if generator is None:
        return "❌ Error: Model not loaded."

    prompt = f"""
    A student scored {score}/10 on a recent test.
    Provide a friendly, personalized feedback message including suggestions to improve further.
    """
    output = generator(prompt)
    return output[0]["generated_text"]

# Step 4: Gradio Interface
def run_all(subject, score, num_questions):
    quiz = generate_quiz(subject, score, num_questions)
    feedback = generate_feedback(score)
    return quiz, feedback

interface = gr.Interface(
    fn=run_all,
    inputs=[
        gr.Textbox(label="Enter Topic (e.g., Algebra)"),
        gr.Slider(0, 10, step=1, label="Score (out of 10)"),
        gr.Slider(1, 10, step=1, label="Number of Questions")
    ],
    outputs=[
        gr.Textbox(label="Generated Quiz", show_copy_button=True),
        gr.Textbox(label="Personalized Feedback", show_copy_button=True)
    ],
    title="EduTutor AI – Personalized Learning & Assessment System",
    description="AI-powered quiz and feedback generator using IBM Granite LLM"
)

interface.launch(debug=True)
