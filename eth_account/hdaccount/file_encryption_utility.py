
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, salt_length=16, iterations=100000):
        self.salt_length = salt_length
        self.iterations = iterations
        self.backend = default_backend()

    def derive_key(self, password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return kdf.derive(password.encode())

    def encrypt_file(self, input_path, output_path, password):
        salt = os.urandom(self.salt_length)
        key = self.derive_key(password, salt)
        
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as f_in:
            plaintext = f_in.read()
        
        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        with open(output_path, 'wb') as f_out:
            f_out.write(salt + iv + ciphertext)
        
        return True

    def decrypt_file(self, input_path, output_path, password):
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length+16]
        ciphertext = data[self.salt_length+16:]
        
        key = self.derive_key(password, salt)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]
        
        with open(output_path, 'wb') as f_out:
            f_out.write(plaintext)
        
        return True

def test_encryption():
    test_data = b"This is a secret message that needs encryption."
    test_file = "test_original.txt"
    encrypted_file = "test_encrypted.bin"
    decrypted_file = "test_decrypted.txt"
    
    with open(test_file, 'wb') as f:
        f.write(test_data)
    
    encryptor = FileEncryptor()
    password = "secure_password_123"
    
    encryptor.encrypt_file(test_file, encrypted_file, password)
    encryptor.decrypt_file(encrypted_file, decrypted_file, password)
    
    with open(decrypted_file, 'rb') as f:
        result = f.read()
    
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    
    return test_data == result

if __name__ == "__main__":
    success = test_encryption()
    print(f"Encryption/decryption test: {'PASSED' if success else 'FAILED'}")