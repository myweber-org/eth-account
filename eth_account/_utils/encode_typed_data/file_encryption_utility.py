import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64

class SimpleFileEncryptor:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self.key = key
        self.backend = default_backend()

    def encrypt_file(self, input_path: str, output_path: str):
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        padder = padding.PKCS7(128).padder()
        
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(iv)
            
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

    def decrypt_file(self, input_path: str, output_path: str):
        with open(input_path, 'rb') as f_in:
            iv = f_in.read(16)
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
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

def generate_key_from_password(password: str, salt: bytes = None) -> bytes:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.hashes import SHA256
    
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file>")
        sys.exit(1)
    
    action = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    password = input("Enter encryption password: ")
    key = generate_key_from_password(password)
    
    encryptor = SimpleFileEncryptor(key)
    
    if action == "encrypt":
        encryptor.encrypt_file(input_file, output_file)
        print(f"File encrypted successfully: {output_file}")
    elif action == "decrypt":
        encryptor.decrypt_file(input_file, output_file)
        print(f"File decrypted successfully: {output_file}")
    else:
        print("Invalid action. Use 'encrypt' or 'decrypt'")