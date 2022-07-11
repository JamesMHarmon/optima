import json
import os

class JSONFile:
    def __init__(self, path):
        self._path = path
        
    def load_or_save_defaults(self, default):
        path = self._path

        if not os.path.isfile(path):
            with open(path, 'w') as f:
                json.dump(default, f, indent = 4)

        with open(path, 'r') as f:
            summary = json.load(f)

        return summary
    
    def load(self):
        with open(self._path, 'r') as f:
            summary = json.load(f)
            
        return summary

    def save_merge(self, merge):
        path = self._path

        with open(path, 'r') as f:
            data = json.load(f)

        print('data', data)    
        
        with open(path, 'w') as f:
            for key, val in merge.items():
                data[key] = val

            json.dump(data, f, indent = 4)
