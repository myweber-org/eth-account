import os
from cryptography.fernet import Fernet

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
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(file_path + '.enc', 'wb') as encrypted_file:
        encrypted_file.write(encrypted)
    os.remove(file_path)
    return file_path + '.enc'

def decrypt_file(encrypted_path, key):
    fernet = Fernet(key)
    with open(encrypted_path, 'rb') as file:
        encrypted = file.read()
    decrypted = fernet.decrypt(encrypted)
    original_path = encrypted_path.replace('.enc', '')
    with open(original_path, 'wb') as file:
        file.write(decrypted)
    os.remove(encrypted_path)
    return original_path

def encrypt_message(message, key):
    fernet = Fernet(key)
    return fernet.encrypt(message.encode())

def decrypt_message(encrypted_message, key):
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_message).decode()

def main():
    print("File Encryption Utility")
    print("1. Generate new key")
    print("2. Encrypt file")
    print("3. Decrypt file")
    print("4. Encrypt message")
    print("5. Decrypt message")
    
    choice = input("Select option: ")
    
    if choice == '1':
        key = generate_key()
        save_key(key)
        print(f"Key generated and saved: {key.decode()}")
    
    elif choice == '2':
        key = load_key()
        file_path = input("Enter file path to encrypt: ")
        if os.path.exists(file_path):
            result = encrypt_file(file_path, key)
            print(f"File encrypted: {result}")
        else:
            print("File not found")
    
    elif choice == '3':
        key = load_key()
        file_path = input("Enter encrypted file path: ")
        if os.path.exists(file_path):
            result = decrypt_file(file_path, key)
            print(f"File decrypted: {result}")
        else:
            print("File not found")
    
    elif choice == '4':
        key = load_key()
        message = input("Enter message to encrypt: ")
        encrypted = encrypt_message(message, key)
        print(f"Encrypted message: {encrypted.decode()}")
    
    elif choice == '5':
        key = load_key()
        message = input("Enter encrypted message: ")
        try:
            decrypted = decrypt_message(message.encode(), key)
            print(f"Decrypted message: {decrypted}")
        except:
            print("Decryption failed - invalid key or message")

if __name__ == "__main__":
    main()