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
    os.remove("test_decrypted.txt")