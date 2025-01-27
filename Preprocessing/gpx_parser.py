import csv
import os
from xml.etree import ElementTree as ET
from datetime import timedelta
from geopy.distance import geodesic
import gpxpy
import gpxpy.gpx


class Run:
    def __init__(self):
        self.moving_time = 0.0
        self.distance_km = 0.0
        self.distance_miles = 0.0
        self.pace_per_km = 0.0
        self.pace_per_mile = 0.0
        self.mph = 0.0
        # self.average_heart_rate = 0.0
        self.total_elevation_gain_m = 0.0
        self.start_datetime = None


def create_run_from_gpx(file_path):
    run = Run()

    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    total_distance = 0.0
    moving_time = timedelta(0)
    total_elevation_gain = 0.0
    heart_rates = []
    start_datetime = None

    # split_distance_km = 1.0  # Default split size in kilometers
    # split_distance_mile = 1.0  # Default split size in miles
    km_split_distance_accum = 0.0
    mile_split_distance_accum = 0.0
    km_split_time_accum = timedelta(0)
    mile_split_time_accum = timedelta(0)

    for track in gpx.tracks:
        for segment in track.segments:
            for i in range(len(segment.points) - 1):
                point1 = segment.points[i]
                point2 = segment.points[i + 1]

                # Record start time
                if not start_datetime and point1.time:
                    start_datetime = point1.time

                # Check time difference to skip pauses
                if point1.time and point2.time:
                    time_diff = point2.time - point1.time
                    if time_diff.total_seconds() > 13:
                        continue
                    moving_time += time_diff
                    km_split_time_accum += time_diff
                    mile_split_time_accum += time_diff

                # Calculate distance
                coord1 = (point1.latitude, point1.longitude)
                coord2 = (point2.latitude, point2.longitude)
                distance_km = geodesic(coord1, coord2).kilometers
                distance_miles = distance_km * 0.621371

                total_distance += distance_km
                km_split_distance_accum += distance_km
                mile_split_distance_accum += distance_miles

                # Calculate elevation gain
                if point2.elevation and point1.elevation:
                    elevation_diff = point2.elevation - point1.elevation
                    if elevation_diff > 0:
                        total_elevation_gain += elevation_diff

                # Extract heart rate directly from extensions
                # for extension in point1.extensions:
                #     hr = extension.find(".//gpxtpx:hr", namespaces={"gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1"})
                #     if hr is not None:
                #         heart_rates.append(int(hr.text))

                # # Check if a kilometer split is completed
                # if km_split_distance_accum >= split_distance_km:
                #     km_splits.append((km_split_distance_accum, km_split_time_accum))
                #     km_split_distance_accum = 0.0
                #     km_split_time_accum = timedelta(0)
                #
                # # Check if a mile split is completed
                # if mile_split_distance_accum >= split_distance_mile:
                #     mile_splits.append((mile_split_distance_accum, mile_split_time_accum))
                #     mile_split_distance_accum = 0.0
                #     mile_split_time_accum = timedelta(0)

    # Handle final splits if they are incomplete
    # if km_split_distance_accum > 0:
    #     km_splits.append((km_split_distance_accum, km_split_time_accum))
    # if mile_split_distance_accum > 0:
    #     mile_splits.append((mile_split_distance_accum, mile_split_time_accum))

    # Convert total distance to miles
    total_distance_miles = total_distance * 0.621371

    # Convert moving time to total minutes
    total_minutes = moving_time.total_seconds() / 60

    # Calculate paces
    pace_per_km = total_minutes / total_distance if total_distance > 0 else 0
    pace_per_mile = total_minutes / total_distance_miles if total_distance_miles > 0 else 0

    # Calculate average speed in mph
    avg_speed_mph = total_distance_miles / (moving_time.total_seconds() / 3600) if moving_time.total_seconds() > 0 else 0

    # Calculate average heart rate
    # avg_heart_rate = sum(heart_rates) / len(heart_rates) if heart_rates else 0

    # Populate the Run object
    run.moving_time = moving_time
    run.distance_km = total_distance
    run.distance_miles = total_distance_miles
    run.pace_per_km = pace_per_km
    run.pace_per_mile = pace_per_mile
    run.mph = avg_speed_mph
    # run.average_heart_rate = avg_heart_rate
    run.total_elevation_gain_m = total_elevation_gain
    run.start_datetime = start_datetime
    # run.km_splits = km_splits
    # run.mile_splits = mile_splits

    return run


def print_run_report(run):
    print(f"Moving Time: {run.moving_time}")
    print(f"Distance: {run.distance_km:.2f} km, {run.distance_miles:.2f} miles")
    print(f"Pace: {run.pace_per_km:.2f} min/km, {run.pace_per_mile:.2f} min/mile")
    print(f"Speed: {run.mph:.2f} mph")
    print(f"Average Heart Rate: {run.average_heart_rate}")
    print(f"Elevation Gain: {run.total_elevation_gain_m:.2f} meters")
    print(f"Start Time: {run.start_datetime}")

    print("\nKilometer Splits:")
    for idx, (distance, time) in enumerate(run.km_splits, start=1):
        print(f"  Split {idx}: {distance:.2f} km in {time}")

    print("\nMile Splits:")
    for idx, (distance, time) in enumerate(run.mile_splits, start=1):
        print(f"  Split {idx}: {distance:.2f} miles in {time}")


def iterate_folder():
    folder_path = "../Data/activities_gpx"

    # Gather all file paths in the folder
    file_paths = [os.path.join(root, file)
                  for root, _, files in os.walk(folder_path)
                  for file in files if file.endswith(".gpx")]

    total_runs = len(file_paths)  # Total number of runs to process
    runs = []

    # Process each GPX file
    for idx, file_path in enumerate(file_paths, start=1):
        print(f"Processing run {idx} of {total_runs}...")
        run = create_run_from_gpx(file_path)  # Assuming create_run_from_gpx returns a Run object
        run_id = f"run_{idx:04d}"  # Unique run ID, e.g., "run_0001", "run_0002"
        run_data = {
            "runId": run_id,
            "moving_time": str(run.moving_time),
            "distance_km": round(run.distance_km, 2),
            "distance_miles": round(run.distance_miles, 2),
            "pace_per_km": round(run.pace_per_km, 2),
            "pace_per_mile": round(run.pace_per_mile, 2),
            "mph": round(run.mph, 2),
            # "average_heart_rate": round(run.average_heart_rate, 4),
            "elevation_gain_m": round(run.total_elevation_gain_m, 2),
            "start_datetime": run.start_datetime.strftime("%Y-%m-%d %H:%M:%S") if run.start_datetime else "N/A"
        }
        runs.append(run_data)


    # Export to CSV
    output_csv = "all_runs.csv"
    with open(output_csv, mode="w", newline="") as csv_file:
        fieldnames = ["runId", "moving_time", "distance_km", "distance_miles",
                      "pace_per_km", "pace_per_mile", "mph",
                      "elevation_gain_m", "start_datetime"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(runs)

    print(f"Runs successfully exported to {output_csv}")

iterate_folder()