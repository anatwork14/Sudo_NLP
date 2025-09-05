import json

def loader(file_path: str, encoding: str = "utf-8") -> str:
    try:
        with open(file_path, encoding=encoding) as f:
            data = json.load(f)
        print('✅ Load Data')
        docs = [d["content"] for d in data]
        return docs
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {file_path}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            f"Unable to decode file at {file_path} with encoding '{encoding}'. "
            "Try a different encoding (e.g., 'utf-16')."
        )
    
