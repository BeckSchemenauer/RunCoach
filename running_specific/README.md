This is used for analyzing or extracting running specific data.

Analysis:
    Two random forest models built to predict average and max HR. Meant to estimate a relative effort if not given by user.

InfoExtractor:
    activity_details:
        Extracts activity specific info such as: time, distance, pace, etc.

    brand_NER:
        Extracts brand names (WIP)

Preprocessing:
    Assembled data from my Strava to build the HR based RF models.