import json
import random
import sys

import nltk
import numpy as np

# Generate 50 random distances with a normal distribution
distances = np.random.normal(loc=7, scale=3, size=100)
# Clip values between 0 and 20
distances = np.clip(distances, 0, 30)
# Convert to string for easy manipulation or use
distances_str = [str(int(distance)) for distance in distances]

# Sentence templates with placeholders
categories = {
    "activity": [
        "I _method_ a _distance_ _unit_ _activity_ for _time_ _time-unit_ .",
        "I _activity_ for _distance_ _unit_ at a pace of _pace_ minutes per _pace-unit_ .",
        "I _method_ a _distance_ _unit_ _activity_ in _time_ minutes .",
        "I _method_ for _time_ _time-unit_ _for_ _distance_ _unit_ .",
        "I _method_ for _time_ _time-unit_ at a _pace_ minutes per _pace-unit_ pace .",
        "I _method_ for _time_ _time-unit_ at a _pace_ minute pace .",
        "I _method_ for a _distance_ _unit_ _activity_ for _time_ _time-unit_ .",
        "It took me _time_ _time-unit_ .",
        "It was at _pace_ minute per _pace-unit_ .",
        "It was at a _pace_ minute per _pace-unit_ pace .",
        "It was _distance_ _unit_ .",
        "I _method_ for _distance_ _unit_ .",
        "I _method_ for _distance_ _unit_ .",
        "It was _distance_ _unit_ .",
    ],
    "modify": [
        "How can I _modify_ my _attribute_ ?",
        "What are some ways to _modify_ my _attribute_ ?",
        "How can I _modify_ my _exercise-type_ ?",
        "What should I do to _modify_ my _attribute_ ?",
        "How do I _modify_ my _attribute_ during runs ?"
    ],
    "validate": [
        "Is this a _describer_ _running-gear_ ?",
        "Are these _describer_ _running-gear-plural_ ?",
    ],
    "generate": [
        "_generative-word_ a _exercise-type_ _plan_ for a _event_ .",
        "Can you _generative-word_ a _exercise-type_ _plan_ for me ?",
        "_generative-word_ a _plan_ for a _event_ .",
        "_generative-word_ a _plan_ for _exercise-type_ ."
    ],
    "recommend": [
        "_recommender_ _describer_ _running-gear_ .",
        "_recommender_ _running-gear_ .",
        "_recommender_ a pair of _running-gear_ for _event_ .",
        "_recommender_ _describer_ _running-gear_ for _event_ training .",
        "_recommender_ a _running-gear_ for improving _attribute_ .",
        "_recommender_ a good _running-gear_ for _event_ runners ."
    ],
    "train": [
        "How can I _train_ for a _event_ ?",
        "What is the best way to _train_ for the _event_ ?",
        "Give me _train_ tips for a _event_ race .",
        "What is the recommended way to _train_ for a _event_ ?"
    ],
    "track": [
        "How do I _track_ my _attribute_ over time ?",
        "What is the best way to _track_ my _pace_ ?",
        "How can I _track_ my progress in _event_ training ?",
        "Can you help me _track_ my running performance ?",
        "How do I _track_ my training schedule for a _event_ ?"
    ]
}

# Word replacement dictionary
replacement_dict = {
    "_method_": ["ran"],
    "_activity_": ["run", "run", "run", "run", "run", "swim", "cycle", "bike", "ride", "jog", "hike", "lift"],
    "_distance_": distances_str,
    "_unit_": ["mile", "kilometer", "meter", "miles", "kilometers", "meters"],
    "_time_": [str(5 * i) for i in range(1, 30)],  # Time intervals in multiples of 5
    "_pace_": [f"{i} : 00" for i in range(2, 10)] + [f"{i} : 30" for i in range(2, 10)],  # pace formats
    "_pace-unit_": ["mile", "kilometer"],
    "_time-unit_": ["minutes", "hours"],
    "_for_": ["for", "over", "covering"],
    "_modify_": ["improve", "better", "increase", "decrease", "raise", "lower", "adjust", "enhance"],
    "_attribute_": ["pace", "speed", "endurance", "strength", "heart", "rate", "stamina", "recovery", "agility"],
    "_running-gear_": ["shoes", "watch", "spikes", "monitor", "bottle", "socks", "belt"],
    "_running-gear-plural_": ["shoes", "watches", "spikes", "monitors", "bottles", "socks", "belts"],
    "_describer_": ["good", "bad", "useful", "essential", "necessary", "comfortable", "lightweight", "durable", "reliable", "expensive", "affordable"],
    "_generative-word_": ["Create", "Give", "Generate", "Suggest", "Formulate", "Provide", "Design"],
    "_exercise-type_": ["training", "running", "lifting", "swimming", "plyometric", "interval", "crossfit", "endurance"],
    "_plan_": ["plan", "schedule", "regime", "routine", "program", "strategy", "calendar"],
    "_event_": ["400 m", "400 meter", "800 m", "800 meter", "1500 m", "1500 meter", "3 k", "5 k", "10 k", "half marathon", "full marathon"],
    "_recommender_": ["Give me", "What is", "What are", "Suggest a", "Recommend me", "Provide a", "Tell me"],
    "_train_": ["train", "prepare", "practice", "condition", "workout", "build", "focus"],
    "_track_": ["track", "monitor", "record", "measure", "check", "log", "analyze"]
}

# Placeholder-to-label mapping
placeholder_labels = {
    "_method_": "method",
    "_activity_": "activity",
    "_distance_": "distance",
    "_unit_": "unit",
    "_time_": "time",
    "_pace_": "pace",
    "_pace-unit_": "pace-unit",
    "_time-unit_": "time-unit",
    "_for_": "none",
    "_event_": "event",
    "_event_attribute_": "event-attribute",
    "_recommender_": "none",  # so I can add one more none token since they're all two words
    "_modify_": "modify",
    "_attribute_": "attribute",
    "_running-gear_": "running-gear",
    "_running-gear-plural_": "running-gear",
    "_describer_": "describer",
    "_generative-word_": "generative-word",
    "_exercise-type_": "exercise-type",
    "_train_": "train",
    "_track_": "track"
}

# Function to replace placeholders in a sentence
def replace_placeholders(sentence, replacements):
    for key, values in replacements.items():
        sentence = sentence.replace(key, random.choice(values))
    return sentence

# Generate 80 sentences with labels
outputs = []

for _ in range(4096):
    # List of categories with weights (activity has 3x weight)
    category_list = list(categories.keys())
    weights = [5 if category == "activity" else 1 for category in category_list]

    # Randomly pick a category based on weights
    chosen_category = random.choices(category_list, weights=weights, k=1)[0]

    # Select a random sentence template from the chosen category
    template = random.choice(categories[chosen_category])

    # Extract labels based on placeholders in the template
    labels = []
    for token in template.split():
        if token in placeholder_labels:
            label = placeholder_labels[token]
            labels.append(label)
            if token == "_pace_":
                labels.append("none")
                labels.append(label)
            if token == "_recommender_":
                labels.append("none")
            if token == "_event_":
                labels.append("event")
        else:
            labels.append("none")  # Assign 'none' to static words

    # Replace placeholders with actual words
    generated_sentence = replace_placeholders(template, replacement_dict)

    tokens = nltk.word_tokenize(generated_sentence)

    if len(tokens) != len(labels):
        print(tokens)
        print(labels)
        sys.exit("Mismatch in number of tokens and labels")

    # Store the result
    outputs.append({
        "tokens": tokens,
        "labels": labels
    })

# Write the generated sentences and labels to a JSON file
output_file = "all_purpose.json"
with open(output_file, "w") as f:
    json.dump(outputs, f, indent=4)

print(f"Generated activity sentences with labels: {output_file}")
