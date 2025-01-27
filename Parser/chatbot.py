from Parser.tagger import load_all_purpose_model, predict_tags, activity_tagger
from Parser.intent_classifier import load_intent_model, classify_intent
from Parser.classes import User, Activity, Input


# Chatbot function
def chatbot():
    print("Hello! I am your chatbot. Type 'exit' to quit.")

    user = User()

    # Load models
    tagging_tokenizer, tagging_model, label_encoder, device = load_all_purpose_model()
    intent_tokenizer, intent_classifier_model = load_intent_model("QuestionClassifier/intent_model")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        predicted_tags = predict_tags(user_input, tagging_tokenizer, tagging_model, label_encoder, device)
        intent = classify_intent(user_input, intent_tokenizer, intent_classifier_model)

        new_activity = activity_tagger(user_input, predicted_tags)
        if new_activity:
            user.add_activity(new_activity)
        input_instance = Input(activity=new_activity, intent=intent, gear=None, other=None)

        print(f"{intent}\n{input_instance.activity}")


# Run the chatbot
chatbot()
