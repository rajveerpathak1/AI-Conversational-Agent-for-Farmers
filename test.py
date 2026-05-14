from chatbot.chain import ask_question

while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    response = ask_question(query)

    print("\nBot:\n")
    print(response["answer"])

    print("\nSources:")

    for source in response["sources"]:
        print("-", source)