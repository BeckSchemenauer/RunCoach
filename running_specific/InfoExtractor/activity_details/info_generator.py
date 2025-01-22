import json
import random
import numpy as np

# Sentence templates with placeholders
categories = {
    "activity": [
        "I _method_ a _distance_ _unit_ _activity_ for _time_ _time-unit_",
        "I _activity_ for _distance_ _unit_ at a pace of _pace_ minutes per _pace-unit_",
        "I _method_ a _distance_ _unit_ _activity_ in _time_ minutes",
        "I _method_ for _time_ _time-unit_ _for_ _distance_ _unit_",
        "I _method_ for _time_ _time-unit_ at a _pace_ minutes per _pace-unit_ pace",
        "I _method_ for _time_ _time-unit_ at a _pace_ minute pace",
        "I _method_ for a _distance_ _unit_ _activity_ for _time_ _time-unit_.",
        "It took me _time_ _time-unit_.",
        "It was at _pace_ minute per _pace-unit_.",
        "It was at a _pace_ minute per _pace-unit_ pace.",
        "It was _distance_ _unit_.",
        "I ran for _distance_ _unit_.",
        "I ran for _distance_ _unit_.",
        "It was _distance_ _unit_.",
    ]
}

# Generate 50 random distances with a normal distribution
distances = np.random.normal(loc=7, scale=3, size=50)
# Clip values between 0 and 20
distances = np.clip(distances, 0, 20)
# Round to the nearest integer or .5
distances = np.round(distances * 2) / 2  # Multiply by 2, round, then divide by 2 to get .0 or .5
# Convert to string for easy manipulation or use
distances_str = [str(distance) for distance in distances]

# Word replacement dictionary
replacement_dict = {
    "_method_": ["did", "went", "completed", "finished", "ran", "swam", "cycled"],
    "_activity_": ["run", "swim", "cycle", "bike", "ride", "jog", "hike", "lifted"],
    "_distance_": distances_str,
    "_unit_": ["mile", "kilometer", "meter", "miles", "kilometers", "meters"],
    "_time_": [str(5 * i) for i in range(1, 30)],  # Time intervals in multiples of 5
    "_pace_": [f"{i}:00" for i in range(2, 10)] + [f"{i}:30" for i in range(2, 10)],  # pace formats
    "_pace-unit_": ["mile", "kilometer"],
    "_time-unit_": ["minutes", "hours"],
    "_for_": ["for", "over", "covering"]
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
    "_for_": "none"  # 'none' as it doesn't need a specific label
}

# Function to replace placeholders in a sentence
def replace_placeholders(sentence, replacements):
    for key, values in replacements.items():
        sentence = sentence.replace(key, random.choice(values))
    return sentence

# Generate 80 sentences with labels
outputs = []

for _ in range(96):
    # Pick a random template
    template = random.choice(categories["activity"])

    # Extract labels based on placeholders in the template
    labels = []
    for token in template.split():
        if token in placeholder_labels:
            labels.append(placeholder_labels[token])
        else:
            labels.append("none")  # Assign 'none' to static words

    # Replace placeholders with actual words
    generated_sentence = replace_placeholders(template, replacement_dict)

    # Store the result
    outputs.append({
        "sentence": generated_sentence,
        "labels": labels
    })

# Write the generated sentences and labels to a JSON file
output_file = "generated_activity_sentences.json"
with open(output_file, "w") as f:
    json.dump(outputs, f, indent=4)

print(f"Generated activity sentences with labels: {output_file}")
