import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import hashlib

def derive_key(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)

def encrypt_file(input_path, output_path, password):
    salt = os.urandom(16)
    key = derive_key(password, salt)
    iv = os.urandom(16)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        f_out.write(salt + iv)
        
        while True:
            chunk = f_in.read(4096)
            if not chunk:
                break
            padded_data = padder.update(chunk)
            encrypted_chunk = encryptor.update(padded_data)
            f_out.write(encrypted_chunk)
        
        final_padded = padder.finalize()
        final_encrypted = encryptor.update(final_padded) + encryptor.finalize()
        f_out.write(final_encrypted)

def decrypt_file(input_path, output_path, password):
    with open(input_path, 'rb') as f_in:
        salt = f_in.read(16)
        iv = f_in.read(16)
        key = derive_key(password, salt)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        
        with open(output_path, 'wb') as f_out:
            while True:
                chunk = f_in.read(4096)
                if not chunk:
                    break
                decrypted_chunk = decryptor.update(chunk)
                unpadded_data = unpadder.update(decrypted_chunk)
                f_out.write(unpadded_data)
            
            final_decrypted = decryptor.finalize()
            final_unpadded = unpadder.update(final_decrypted) + unpadder.finalize()
            f_out.write(final_unpadded)

if __name__ == "__main__":
    test_data = b"Test data for encryption and decryption."
    with open("test.txt", "wb") as f:
        f.write(test_data)
    
    encrypt_file("test.txt", "test.enc", "secure_password")
    decrypt_file("test.enc", "test_decrypted.txt", "secure_password")
    
    with open("test_decrypted.txt", "rb") as f:
        decrypted = f.read()
    
    print("Original:", test_data)
    print("Decrypted:", decrypted)
    print("Match:", test_data == decrypted)
    
    os.remove("test.txt")
    os.remove("test.enc")
    os.remove("test_decrypted.txt")import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([b ^ key for b in data])

def process_file(input_path, output_path, key):
    """Process a file for encryption or decryption."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        processed_data = xor_cipher(data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"Operation completed. Output saved to: {output_path}")
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
            raise ValueError("Key out of range")
    except ValueError as e:
        print(f"Invalid key: {e}")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    if os.path.exists(output_file):
        response = input(f"Output file '{output_file}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    success = process_file(input_file, output_file, key)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()