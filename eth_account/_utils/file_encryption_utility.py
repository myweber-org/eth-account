import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([b ^ key for b in data])

def process_file(input_path, output_path, key):
    """Encrypt or decrypt a file."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        processed_data = xor_cipher(data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_utility.py <input_file> <output_file> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        key = int(sys.argv[3])
        if not 0 <= key <= 255:
            raise ValueError("Key must be between 0 and 255")
    except ValueError as e:
        print(f"Invalid key: {e}")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)
    
    if process_file(input_file, output_file, key):
        print("Operation completed successfully.")
    else:
        print("Operation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.password = password.encode('utf-8')
        self.salt = get_random_bytes(16)
        self.key = PBKDF2(self.password, self.salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        if output_path is None:
            output_path = input_path + '.enc'
        
        with open(output_path, 'wb') as f:
            f.write(self.salt + iv + ciphertext)
        
        return output_path

    def decrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]
        
        key = PBKDF2(self.password, salt, dkLen=32, count=1000000)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path

    def calculate_hash(self, file_path, algorithm='sha256'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()

def example_usage():
    encryptor = FileEncryptor("secure_password_123")
    
    test_content = b"This is a secret message that needs encryption."
    test_file = "test_secret.txt"
    
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    print(f"Original file created: {test_file}")
    print(f"Original hash: {encryptor.calculate_hash(test_file)}")
    
    encrypted_file = encryptor.encrypt_file(test_file)
    print(f"Encrypted file: {encrypted_file}")
    print(f"Encrypted hash: {encryptor.calculate_hash(encrypted_file)}")
    
    decrypted_file = encryptor.decrypt_file(encrypted_file)
    print(f"Decrypted file: {decrypted_file}")
    print(f"Decrypted hash: {encryptor.calculate_hash(decrypted_file)}")
    
    with open(decrypted_file, 'rb') as f:
        decrypted_content = f.read()
    
    print(f"Content matches: {decrypted_content == test_content}")
    
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    print("Test files cleaned up")

if __name__ == "__main__":
    example_usage()