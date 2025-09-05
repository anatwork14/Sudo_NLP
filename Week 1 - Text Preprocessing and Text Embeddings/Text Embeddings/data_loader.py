def loader(file_path: str, encoding: str = "utf-16") -> str:
    try:
        with open(file_path, encoding=encoding) as f:
            data = f.read()
        print('✅ Load Data')
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {file_path}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            f"Unable to decode file at {file_path} with encoding '{encoding}'. "
            "Try a different encoding (e.g., 'utf-8')."
        )
