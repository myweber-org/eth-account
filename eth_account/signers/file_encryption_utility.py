
import os
import sys

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = xor_cipher(plaintext, key)
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        print(f"Encryption successful. Output saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Encryption failed: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        key = int(sys.argv[4])
        if not (0 <= key <= 255):
            raise ValueError
    except ValueError:
        print("Error: Key must be an integer between 0 and 255")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    if operation == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Error: Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.key = hashlib.sha256(password.encode()).digest()
    
    def encrypt_file(self, input_path, output_path=None):
        if output_path is None:
            output_path = input_path + '.enc'
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        with open(output_path, 'wb') as f:
            f.write(iv + ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path, output_path=None):
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path
    
    def encrypt_string(self, text):
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(text.encode(), AES.block_size))
        return b64encode(iv + ciphertext).decode()
    
    def decrypt_string(self, encrypted_text):
        data = b64decode(encrypted_text)
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return plaintext.decode()

def main():
    encryptor = FileEncryptor("secure_password123")
    
    test_text = "This is a secret message"
    encrypted = encryptor.encrypt_string(test_text)
    decrypted = encryptor.decrypt_string(encrypted)
    
    print(f"Original: {test_text}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    
    test_file = "test_data.txt"
    with open(test_file, 'w') as f:
        f.write("Sensitive file content\nMultiple lines\nEnd of file")
    
    encrypted_file = encryptor.encrypt_file(test_file)
    decryptor = FileEncryptor("secure_password123")
    decrypted_file = decryptor.decrypt_file(encrypted_file)
    
    with open(decrypted_file, 'r') as f:
        print(f"Decrypted file content:\n{f.read()}")
    
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)

if __name__ == "__main__":
    main()