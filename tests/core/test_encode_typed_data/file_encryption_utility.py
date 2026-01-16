from cryptography.fernet import Fernet
import os
import sys

def generate_key():
    return Fernet.generate_key()

def save_key(key, key_file='secret.key'):
    with open(key_file, 'wb') as file:
        file.write(key)

def load_key(key_file='secret.key'):
    return open(key_file, 'rb').read()

def encrypt_file(file_path, key):
    fernet = Fernet(key)
    with open(file_path, 'rb') as file:
        original_data = file.read()
    encrypted_data = fernet.encrypt(original_data)
    with open(file_path + '.encrypted', 'wb') as file:
        file.write(encrypted_data)
    return file_path + '.encrypted'

def decrypt_file(encrypted_file_path, key):
    fernet = Fernet(key)
    with open(encrypted_file_path, 'rb') as file:
        encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    original_file_path = encrypted_file_path.replace('.encrypted', '.decrypted')
    with open(original_file_path, 'wb') as file:
        file.write(decrypted_data)
    return original_file_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_encryption_utility.py <command> [arguments]")
        print("Commands:")
        print("  generate_key [key_file]")
        print("  encrypt <file> [key_file]")
        print("  decrypt <encrypted_file> [key_file]")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'generate_key':
        key_file = sys.argv[2] if len(sys.argv) > 2 else 'secret.key'
        key = generate_key()
        save_key(key, key_file)
        print(f"Key generated and saved to {key_file}")

    elif command == 'encrypt':
        if len(sys.argv) < 3:
            print("Error: File to encrypt not specified")
            sys.exit(1)
        file_to_encrypt = sys.argv[2]
        key_file = sys.argv[3] if len(sys.argv) > 3 else 'secret.key'
        key = load_key(key_file)
        encrypted_file = encrypt_file(file_to_encrypt, key)
        print(f"File encrypted: {encrypted_file}")

    elif command == 'decrypt':
        if len(sys.argv) < 3:
            print("Error: File to decrypt not specified")
            sys.exit(1)
        file_to_decrypt = sys.argv[2]
        key_file = sys.argv[3] if len(sys.argv) > 3 else 'secret.key'
        key = load_key(key_file)
        decrypted_file = decrypt_file(file_to_decrypt, key)
        print(f"File decrypted: {decrypted_file}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()