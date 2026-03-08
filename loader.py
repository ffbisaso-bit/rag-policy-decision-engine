from pathlib import Path


def load_text(filename):
    base_path = Path(__file__).parent
    file_path = base_path / filename
    text = file_path.read_text()
    return text


if __name__ == "__main__":
    text = load_text("../data/company_policy.txt")

    print("Loaded characters:", len(text))
    print("\n--- Preview ---\n")
    print(text[:500])