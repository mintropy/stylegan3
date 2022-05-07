import json
import os
from pathlib import Path


def setup():
    if not os.path.isdir("./data"):
        os.mkdir("data")
    if not os.path.isdir("./data/images"):
        os.mkdir("data/images")
    if not os.path.isfile("./data/data.json"):
        file_name = "data/data.json"
        json_data = {}
        with open(file_name, 'w') as f:
            json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    setup()
