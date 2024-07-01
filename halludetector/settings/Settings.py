import json

class Settings:
    def __init__(self, filename):
        self.filename = filename
        self.settings = self.load_settings()

    def load_settings(self):
        with open(self.filename, 'r') as f:
            return json.load(f)

    def save_settings(self):
        with open(self.filename, 'w') as f:
            json.dump(self.settings, f, indent=2)

    def update_settings(self, field_key, new_value):
        for field in self.settings:
            if field['key'] == field_key:
                field['value'] = new_value
                return
        else:
            raise KeyError(f"Field with '{field_key}' not found")
        
    def get_settings_value(self, key):
        setting = next((item for item in self.settings if item['key'] == key), None)
        if setting:
            return setting.get('value')
        else:
            return None
