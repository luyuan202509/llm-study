import os

class RawText:
    def __init__(self, file_path=None):
        if file_path is None:
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.file_path = os.path.join(parent_path, 'ch05', 'the-verdict.txt')
        else:
            self.file_path = file_path
        self.raw_text = None

    def read(self):
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                self.raw_text = f.read()
            return self.raw_text
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None