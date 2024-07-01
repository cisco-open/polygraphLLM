class DatasetParser:
    def __init__(self):
        self.download_data()
    
    def download_data(self):
        raise NotImplementedError  

    def display(self, offset, limit):
        raise NotImplementedError
    
    def apply_offset_limit(self, data, offset, limit):
        return data[offset: offset + limit]
    