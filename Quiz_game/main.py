from question_model import Question
from data import question_data
from quiz_brain import QuizBrain

question_bank = []

for question in question_data:
    question_text = question["question"]
    question_answer = question["correct_answer"]
    new_question = Question(question_text, question_answer)
    question_bank.append(new_question)

quiz = QuizBrain(question_bank)
while quiz.still_there_is_next_question():
    quiz.next_question()
final_score = quiz.score
question_number = quiz.question_number
print("You've completed the quiz")
print(f"Your final score was:{final_score}/{question_number} ")





