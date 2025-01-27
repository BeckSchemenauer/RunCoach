# Define the activity_tagger function
import nltk
from Parser.classes import Activity


def activity_tagger(sentence, predicted_tags):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)

    # Initialize variables to store the values
    activity = None
    distance = None
    distance_unit = None
    time = None
    pace = None
    pace_unit = None
    elevation = None
    relative_effort = None
    heart_rate = None

    # Loop through predicted tags and assign values to corresponding variables
    for i, tag in enumerate(predicted_tags):
        if tag == 'activity':
            activity = tokens[i]
        elif tag == 'distance':
            distance = tokens[i]
        elif tag == 'unit':
            distance_unit = tokens[i]
        elif tag == 'time':
            time = tokens[i]
        elif tag == 'pace':
            pace = tokens[i]
        elif tag == 'pace-unit':
            pace_unit = tokens[i]

    # Create and return the Activity object with the extracted values
    return Activity(activity, distance, distance_unit, time, pace, pace_unit, elevation, relative_effort, heart_rate)
