import json
import random
import numpy as np

# Sentence templates with placeholders
templates = [
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
    "I ran _distance_ _unit_."
    "It was _distance_ _unit_.",
]

# Generate distances (normal distribution, clipped, rounded to .0 or .5)
distances = [(round(np.clip(d, 0, 20) * 2) / 2) for d in np.random.normal(7, 3, 50)]
print(distances)

# Generate minutes (normal distribution centered on 55, integers)
minutes = [(int(np.clip(d, 1, 120))) for d in np.random.normal(55, 15, 50)]
print(minutes)

hours = [round(np.clip(d, 0, 24) * 2) / 2 for d in np.random.normal(1.5, 0.5, 50)]
print(hours)

# Word replacement dictionary
replacement_dict = {
    "_method_": [("did", 2), ("went", 1), ("completed", 1), ("finished", 1), ("ran", 5), ("swam", 1), ("cycled", 1)],
    "_activity_": [("run", 5), ("swim", 1), ("cycle", 1), ("bike", 1), ("ride", 1), ("jog", 1), ("hike", 1), ("lifted", 1)],
    "_distance_": distances,  # No weights
    "_unit_": [("mile", 3), ("kilometer", 2), ("meter", 1), ("miles", 3), ("kilometers", 2), ("meters", 1)],
    "_time_": None,
    "_pace_": [f"{i}:00" for i in range(2, 10)] + [f"{i}:30" for i in range(2, 10)],  # No weights
    "_pace-unit_": [("mile", 3), ("kilometer", 2)],
    "_time-unit_": [("minutes", 4), ("hours", 1)],
    "_for_": [("for", 3), ("over", 1), ("covering", 1)]
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


# Replaces placeholders with weighted or unweighted choices
def replace_placeholders(sentence, replacements):
    time_value = None  # To store the selected time value for conditional logic
    for key, values in replacements.items():
        if key == "_time_":
            if random.random() < 0.8:
                # Choose a time value and convert to float for comparison
                time_value = float(random.choice(values))
                sentence = sentence.replace(key, str(time_value))
        elif key == "_time-unit_":
            # Apply conditional logic for time units
            if time_value in {0.5, 1, 1.5, 2.5, 3}:
                sentence = sentence.replace(key, "hours")
            else:
                sentence = sentence.replace(key, "minutes")
        elif isinstance(values[0], tuple):  # Check if weights are provided
            items, weights = zip(*values)
            sentence = sentence.replace(key, random.choices(items, weights=weights)[0])
        else:  # Handle cases without weights
            sentence = sentence.replace(key, random.choice(values))
    return sentence

# Generate base set of sentences with labels
outputs = []

# 5 times length of base examples
for _ in range(5):
    for template in templates:
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

# Pick x additional templates to total 192
x = 192 - 5 * len(templates)
print(x)
for _ in range(x):
    # Pick a random template
    template = random.choice(templates)

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
output_file = "test_sentences.json"
with open(output_file, "w") as f:
    json.dump(outputs, f, indent=4)

print(f"Generated activity sentences with labels: {output_file}")
