from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

# Step 1: Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 2: Load model & tokenizer
try:
    model_name = "ibm-granite/granite-3.3-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=700
    )
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {e}")
    generator = None

# Utility function to generate text
def generate_response(prompt):
    if generator is None:
        return "❌ Error: Model not loaded."
    response = generator(prompt)
    return response[0]["generated_text"]

# Functionality 1: Generate Quiz
def generate_quiz(subject: str, score: int, num_questions: int):
    prompt = f"""
You are an expert tutor.

Topic: {subject}
Student Score: {score}/10

Generate {num_questions} multiple-choice questions to help the student's understanding of the topic '{subject}'.

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
    return generate_response(prompt)

# Functionality 2: Feedback Generator
def generate_feedback(score):
    prompt = f"A student scored {score}/10. Provide a friendly, personalized feedback message with suggestions to improve."
    return generate_response(prompt)

# Functionality 3: Recommended Resources
def generate_resources(subject):
    prompt = f"Provide 5 free, high-quality online learning resources (websites, YouTube, courses) to study the topic: {subject}."
    return generate_response(prompt)

# Functionality 4: Summary Notes
def generate_summary_notes(subject):
    prompt = f"Give a beginner-friendly summary of the topic '{subject}' with clear and simple explanation."
    return generate_response(prompt)

# Functionality 5: Adaptive Question Suggestion
def generate_adaptive_question(subject, score):
    difficulty = "easy" if score <= 4 else "medium" if score <= 7 else "hard"
    prompt = f"Generate one {difficulty}-level multiple choice question on the topic: {subject}."
    return generate_response(prompt)

# Functionality 6: Concept-wise MCQ Generation
def generate_concept_questions(subject, concept):
    prompt = f"Generate 3 multiple-choice questions focused on the sub-topic '{concept}' under '{subject}'."
    return generate_response(prompt)

# Functionality 7: Fill in the Blanks
def generate_fill_in_the_blanks(subject):
    prompt = f"""
Generate 5 fill-in-the-blank questions with answers on the topic: '{subject}'.

Format:
Q1: <question with blank>
Answer: <correct word or phrase>

Ensure each blank tests an important concept from the topic.
"""
    return generate_response(prompt)

# Functionality 8: Important Points
def generate_important_points(subject):
    prompt = f"""
List the 7 most important points a beginner should remember when studying the topic: '{subject}'. Use short, clear bullet points.
"""
    return generate_response(prompt)

# Functionality 9: Flashcard Format Output
def generate_flashcards(subject, num_flashcards):
    prompt = f"Generate {num_flashcards} flashcards for the topic '{subject}'. Format each as: Q: <question> A: <answer>"
    return generate_response(prompt)

# Functionality 10: Misconception Correction
def generate_misconceptions(subject):
    prompt = f"""
List common misconceptions students have when learning the topic: '{subject}'. For each one, provide a correct explanation.

Format:
Misconception: <wrong idea>
Correction: <correct understanding>
"""
    return generate_response(prompt)

# Functionality 11: Confidence Score Explanation
def confidence_analysis(score):
    prompt = f"A student scored {score}/10. Analyze their confidence level and suggest how to build stronger understanding in weak areas."
    return generate_response(prompt)

# Functionality 12: Weekly Learning Plan Generator
def generate_study_plan(subject, score):
    prompt = f"A student scored {score}/10 on the topic '{subject}'. Create a personalized 5-day learning plan to improve their understanding."
    return generate_response(prompt)

# Gradio App
def run_all(subject, score, num_questions, concept, flashcard_count):
    quiz = generate_quiz(subject, score, num_questions)
    feedback = generate_feedback(score)
    resources = generate_resources(subject)
    notes = generate_summary_notes(subject)
    adaptive = generate_adaptive_question(subject, score)
    concept_questions = generate_concept_questions(subject, concept)
    fill_blanks = generate_fill_in_the_blanks(subject)
    important_points = generate_important_points(subject)
    flashcards = generate_flashcards(subject, flashcard_count)
    misconceptions = generate_misconceptions(subject)
    confidence = confidence_analysis(score)
    study_plan = generate_study_plan(subject, score)

    return quiz, feedback, resources, notes, adaptive, concept_questions, fill_blanks, important_points, misconceptions, confidence, flashcards, study_plan

interface = gr.Interface(
    fn=run_all,
    inputs=[
        gr.Textbox(label="Topic (e.g., Algebra)"),
        gr.Slider(0, 10, step=1, label="Score (out of 10)"),
        gr.Slider(1, 10, step=1, label="Number of Questions"),
        gr.Textbox(label="Concept Name (e.g., Linear Equations)"),
        gr.Slider(1, 10, step=1, label="Number of Flashcards")
    ],
    outputs=[
        gr.Textbox(label="Generated Quiz", show_copy_button=True),
        gr.Textbox(label="Personalized Feedback", show_copy_button=True),
        gr.Textbox(label="Learning Resources", show_copy_button=True),
        gr.Textbox(label="Summary Notes", show_copy_button=True),
        gr.Textbox(label="Adaptive Question", show_copy_button=True),
        gr.Textbox(label="Concept-Based Questions", show_copy_button=True),
        gr.Textbox(label="Fill in the Blanks", show_copy_button=True),
        gr.Textbox(label="Important Points", show_copy_button=True),
        gr.Textbox(label="Flashcards", show_copy_button=True),
        gr.Textbox(label="Misconception Correction", show_copy_button=True),
        gr.Textbox(label="Confidence Analysis", show_copy_button=True),
        gr.Textbox(label="Weekly Study Plan", show_copy_button=True)
    ],
    title="EduTutor AI – Personalized Learning & Assessment System",
    description="📚 Generate quizzes, feedback, flashcards, study plans, and more using IBM Granite LLM"
)

interface.launch(debug=True)
