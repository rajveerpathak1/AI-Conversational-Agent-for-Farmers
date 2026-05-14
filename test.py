from chatbot.chain import ask_question

while True:

    question = input("Ask: ")

    response = ask_question(question)

    print("\nBot:", response)