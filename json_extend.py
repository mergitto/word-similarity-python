import json

def json_dump(dictionary):
    return json.dumps(dictionary, sort_keys=True)
