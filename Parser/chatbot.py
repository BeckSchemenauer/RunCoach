from transformers import BertTokenizer

from Parser.activity_tagger import load_model, load_vocabulary_and_encoder, predict_tags, activity_tagger
from Parser.intent_classifier import load_intent_model, classify_intent
from Parser.classes import User, Activity, Input


# Chatbot function
def chatbot():
    print("Hello! I am your chatbot. Type 'exit' to quit.")

    user = User()

    # Load models
    vocab, label_encoder = load_vocabulary_and_encoder()
    info_extractor_model = load_model(len(vocab) + 1, len(label_encoder.classes_))
    intent_tokenizer, intent_classifier_model = load_intent_model("QuestionClassifier/intent_model")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        predicted_tags = predict_tags(user_input, info_extractor_model, vocab, label_encoder)
        intent = classify_intent(user_input, intent_tokenizer, intent_classifier_model)

        new_activity = activity_tagger(user_input, predicted_tags)
        user.add_activity(new_activity)
        input_instance = Input(activity=new_activity, intent=intent, gear=None, other=None)

        print(f"{intent}\n{input_instance.activity}")


# Run the chatbot
chatbot()
