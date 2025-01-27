class Run:
    def __init__(self, distance, time, pace, elevation, relative_effort, heart_rate):
        """
        Initializes a Run object.
        :param distance: Distance of the run in kilometers or miles (float)
        :param time: Total time of the run in seconds (float)
        :param pace: Average pace per kilometer or mile in seconds (float)
        :param elevation: Elevation gain during the run in meters or feet (float)
        :param relative_effort: Relative effort of the run (int or float)
        :param heart_rate: Average heart rate during the run in bpm (int or float)
        """
        self.distance = distance
        self.time = time
        self.pace = pace
        self.elevation = elevation
        self.relative_effort = relative_effort
        self.heart_rate = heart_rate


class User:
    def __init__(self, name, gear, height, weight, sex):
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
        self.runs = []  # List to hold Run objects

    def add_run(self, run):
        """
        Adds a Run object to the user's list of runs.
        :param run: Run object to add (Run)
        """
        if isinstance(run, Run):
            self.runs.append(run)
        else:
            raise ValueError("Input must be an instance of the Run class")


class Input:
    def __init__(self, run, intent, gear, other):
        """
        Initializes an Input object.
        :param run: A Run object (Run)
        :param intent: The intent of the input, e.g., 'log_run', 'update_gear' (str)
        :param gear: Dictionary containing gear information with keys 'watch' and 'shoes' (dict)
        :param other: Dictionary containing additional information with keys 'name', 'height', 'weight', 'sex' (dict)
        """
        if not isinstance(run, Run):
            raise ValueError("'run' must be an instance of the Run class")

        self.run = run
        self.intent = intent
        self.gear = gear
        self.other = other

        # Validate other dictionary keys
        required_keys = ['name', 'height', 'weight', 'sex']
        for key in required_keys:
            if key not in self.other:
                raise ValueError(f"Missing required key in 'other': {key}")
