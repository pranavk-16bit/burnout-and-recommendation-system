import json
from pathlib import Path
from cryptography.fernet import Fernet

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "burnout_history.enc"
KEY_PATH = Path("data/secret.key")

DATA_PATH.parent.mkdir(exist_ok=True)


def get_or_create_key():
    """Load the encryption key, or create one if it doesn't exist."""

    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()

    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    return key


def load_history():
    """Return the list of saved history records."""

    if not DATA_PATH.exists():
        return []

    key = get_or_create_key()
    fernet = Fernet(key)

    encrypted = DATA_PATH.read_bytes()
    decrypted = fernet.decrypt(encrypted)

    return json.loads(decrypted.decode())


def save_history(password, data):
    """Add one history record and save the encrypted file."""

    history = load_history()
    history.append(data)

    key = get_or_create_key()
    fernet = Fernet(key)

    encrypted = fernet.encrypt(json.dumps(history).encode())
    DATA_PATH.write_bytes(encrypted)


def delete_all_data(password):
    """Remove all saved history."""

    if DATA_PATH.exists():
        DATA_PATH.unlink()
