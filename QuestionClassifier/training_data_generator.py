import json
import random

# Categories with base questions
categories = {
    "modify": [
        "How can I _modify_ my _attribute_?",
        "What are some ways to _modify_ my _attribute_?",
        "How can I _modify_ my _exercise-type_?",
        "What should I do to _modify_ my _attribute_?",
        "How do I _modify_ my _attribute_ during runs?"
    ],
    "validate": [
        "Is this a _describer_ _running-gear_?",
        "Are these _describer_ _running-gear-plural_?",
    ],
    "generate": [
        "_generative-word_ a _exercise-type_ _plan_ for a _event_.",
        "Can you _generative-word_ a _exercise-type_ _plan_ for me?",
        "_generative-word_ a _plan_ for a _event_.",
        "_generative-word_ a _plan_ for _exercise-type_."
    ],
    "recommend": [
        "_recommender_ _describer_ _running-gear_.",
        "_recommender_ _running-gear_.",
        "_recommender_ a pair of _running-gear_ for _event_.",
        "_recommender_ _describer_ _running-gear_ for _event_ training.",
        "_recommender_ a _running-gear_ for improving _attribute_.",
        "_recommender_ a good _running-gear_ for _event_ runners."
    ],
    "train": [
        "How can I _train_ for a _event_?",
        "What is the best way to _train_ for the _event_?",
        "Give me _train_ tips for a _event_ race.",
        "What is the recommended way to _train_ for a _event_?"
    ],
    "track": [
        "How do I _track_ my _attribute_ over time?",
        "What is the best way to _track_ my _pace_?",
        "How can I _track_ my progress in _event_ training?",
        "Can you help me _track_ my running performance?",
        "How do I _track_ my training schedule for a _event_?"
    ]
}


# Word replacement dictionary
replacement_dict = {
    "_modify_": ["improve", "better", "increase", "decrease", "raise", "lower", "adjust", "enhance"],
    "_attribute_": ["pace", "speed", "endurance", "strength", "heart rate", "stamina", "recovery time", "agility"],
    "_running-gear_": ["pair of shoes", "watch", "pair of spikes", "heart rate monitor", "water bottle", "pair of compression socks", "running belt"],
    "_running-gear-plural_": ["shoes", "watches", "spikes", "heart rate monitors", "water bottles", "compression socks", "running belts"],
    "_describer_": ["good", "bad", "useful", "essential", "necessary", "comfortable", "lightweight", "durable", "reliable", "expensive", "affordable"],
    "_generative-word_": ["Create", "Give me", "Generate", "Suggest", "Formulate", "Provide", "Design"],
    "_exercise-type_": ["training", "running", "lifting", "swimming", "plyometric", "interval training", "crossfit", "endurance training"],
    "_plan_": ["plan", "schedule", "regime", "routine", "program", "strategy", "workout plan", "training calendar"],
    "_event_": ["400m", "800m", "1500m", "3k", "5k", "10k", "half marathon", "marathon", "ultramarathon", "triathlon", "5-mile race", "10-mile race", "half ironman", "ironman"],
    "_recommender_": ["Give me a", "What is a", "What are", "Suggest", "Recommend me a", "Can you provide a", "Tell me the best"],
    "_train_": ["train", "prepare", "get ready", "practice", "condition", "workout", "build stamina", "build strength", "focus on"],
    "_track_": ["track", "monitor", "record", "measure", "check", "log", "keep track of", "analyze"]
}


outputs = {"modify": [],
           "validate": [],
           "generate": [],
           "recommend": [],
           "train": [],
           "track": []}


# Function to replace placeholders in a question
def replace_placeholders(question, replacements):
    for key, values in replacements.items():
        question = question.replace(key, random.choice(values))
    return question


# Replace placeholders and generate 20 random questions for each category
for category in categories:
    prompt_list = categories[category]
    for i in range(40):
        base_question = random.choice(prompt_list)
        modified_question = replace_placeholders(base_question, replacement_dict)
        outputs[category].append(modified_question)

# Write to JSON file
output_file = "labeled_questions_dict.json"
with open(output_file, "w") as f:
    json.dump(outputs, f, indent=4)

print(f"Generated labeled questions: {output_file}")
