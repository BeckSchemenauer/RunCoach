class Activity:
    def __init__(self, activity, distance, time, unit, pace, pace_unit, elevation, relative_effort, heart_rate):
        self.activity = activity
        self.distance = distance
        self.time = time
        self.unit = unit
        self.pace = pace
        self.pace_unit = pace_unit
        self.elevation = elevation
        self.relative_effort = relative_effort
        self.heart_rate = heart_rate

    def __repr__(self):
        return f"Activity(activity={self.activity}, distance={self.distance}, unit={self.unit}, time={self.time}, pace={self.pace}, pace_unit={self.pace_unit}, elevation={self.elevation}, relative_effort={self.relative_effort}, heart_rate={self.heart_rate})"


class User:
    def __init__(self, name=None, gear=None, height=None, weight=None, sex=None, age=None):
        """
        Initializes a User object.
        :param name: Name of the user (str)
        :param gear: Dictionary containing user's gear with keys 'watch' and 'shoes' (dict)
        :param height: Height of the user in centimeters or inches (float)
        :param weight: Weight of the user in kilograms or pounds (float)
        :param sex: Sex of the user ('male', 'female', or other) (str)
        """
        self.name = name
        self.gear = gear
        self.height = height
        self.weight = weight
        self.sex = sex
        self.age = age
        self.runs = []  # List to hold Run objects

    def add_activity(self, activity):
        if isinstance(activity, Activity):
            self.runs.append(activity)
        else:
            raise ValueError("Input must be an instance of the Activity class")


class Input:
    def __init__(self, activity=None, intent=None, gear=None, other=None):
        """
        Initializes an Input object.
        :param activity: An Activity object
        :param intent: The intent of the input, e.g., 'log_run', 'update_gear' (str)
        :param gear: Dictionary containing gear information with keys 'watch' and 'shoes' (dict)
        :param other: Dictionary containing additional information with keys 'name', 'height', 'weight', 'sex' (dict)
        """

        self.activity = activity
        self.intent = intent
        self.gear = gear
        self.other = other
