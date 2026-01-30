import os
from cryptography.fernet import Fernet

def generate_key():
    return Fernet.generate_key()

def save_key(key, key_file='secret.key'):
    with open(key_file, 'wb') as file:
        file.write(key)

def load_key(key_file='secret.key'):
    with open(key_file, 'rb') as file:
        return file.read()

def encrypt_file(input_file, output_file=None, key=None):
    if key is None:
        key = load_key()
    fernet = Fernet(key)
    
    with open(input_file, 'rb') as file:
        original_data = file.read()
    
    encrypted_data = fernet.encrypt(original_data)
    
    if output_file is None:
        output_file = input_file + '.encrypted'
    
    with open(output_file, 'wb') as file:
        file.write(encrypted_data)
    
    return output_file

def decrypt_file(input_file, output_file=None, key=None):
    if key is None:
        key = load_key()
    fernet = Fernet(key)
    
    with open(input_file, 'rb') as file:
        encrypted_data = file.read()
    
    decrypted_data = fernet.decrypt(encrypted_data)
    
    if output_file is None:
        if input_file.endswith('.encrypted'):
            output_file = input_file[:-10]
        else:
            output_file = input_file + '.decrypted'
    
    with open(output_file, 'wb') as file:
        file.write(decrypted_data)
    
    return output_file

def encrypt_string(plaintext, key=None):
    if key is None:
        key = load_key()
    fernet = Fernet(key)
    return fernet.encrypt(plaintext.encode())

def decrypt_string(ciphertext, key=None):
    if key is None:
        key = load_key()
    fernet = Fernet(key)
    return fernet.decrypt(ciphertext).decode()

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <filename> [output_filename]")
        sys.exit(1)
    
    action = sys.argv[1]
    filename = sys.argv[2]
    output_filename = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists('secret.key'):
        key = generate_key()
        save_key(key)
        print("Generated new encryption key")
    
    if action == 'encrypt':
        result = encrypt_file(filename, output_filename)
        print(f"File encrypted: {result}")
    elif action == 'decrypt':
        result = decrypt_file(filename, output_filename)
        print(f"File decrypted: {result}")
    else:
        print("Invalid action. Use 'encrypt' or 'decrypt'")

if __name__ == "__main__":
    main()