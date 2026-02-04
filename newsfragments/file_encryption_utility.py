from cryptography.fernet import Fernet
import os
import base64

class FileEncryptor:
    def __init__(self, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.key = key
        self.cipher = Fernet(self.key)

    def save_key(self, key_file='secret.key'):
        with open(key_file, 'wb') as f:
            f.write(self.key)
        print(f"Key saved to {key_file}")

    def load_key(self, key_file='secret.key'):
        with open(key_file, 'rb') as f:
            self.key = f.read()
        self.cipher = Fernet(self.key)
        print(f"Key loaded from {key_file}")

    def encrypt_file(self, input_file, output_file=None):
        if output_file is None:
            output_file = input_file + '.encrypted'

        with open(input_file, 'rb') as f:
            data = f.read()

        encrypted_data = self.cipher.encrypt(data)

        with open(output_file, 'wb') as f:
            f.write(encrypted_data)

        print(f"File encrypted: {input_file} -> {output_file}")
        return output_file

    def decrypt_file(self, input_file, output_file=None):
        if output_file is None:
            if input_file.endswith('.encrypted'):
                output_file = input_file[:-10]
            else:
                output_file = input_file + '.decrypted'

        with open(input_file, 'rb') as f:
            encrypted_data = f.read()

        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
        except Exception as e:
            print(f"Decryption failed: {e}")
            return None

        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

        print(f"File decrypted: {input_file} -> {output_file}")
        return output_file

    def encrypt_string(self, text):
        if isinstance(text, str):
            text = text.encode()
        encrypted = self.cipher.encrypt(text)
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_string(self, encrypted_text):
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted = self.cipher.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            print(f"String decryption failed: {e}")
            return None

def example_usage():
    encryptor = FileEncryptor()
    encryptor.save_key()

    test_text = "Sensitive information here"
    encrypted_text = encryptor.encrypt_string(test_text)
    print(f"Encrypted text: {encrypted_text}")

    decrypted_text = encryptor.decrypt_string(encrypted_text)
    print(f"Decrypted text: {decrypted_text}")

    test_file = 'test.txt'
    with open(test_file, 'w') as f:
        f.write("This is a test file with secret data.")

    encrypted_file = encryptor.encrypt_file(test_file)
    decryptor = FileEncryptor()
    decryptor.load_key()
    decrypted_file = decryptor.decrypt_file(encrypted_file)

    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    os.remove('secret.key')

if __name__ == "__main__":
    example_usage()import os
from cryptography.fernet import Fernet

class FileEncryptor:
    def __init__(self, key_file='secret.key'):
        self.key_file = key_file
        self.key = None
        self.cipher = None
        
    def generate_key(self):
        self.key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(self.key)
        print(f"Key generated and saved to {self.key_file}")
        return self.key
    
    def load_key(self):
        if not os.path.exists(self.key_file):
            raise FileNotFoundError(f"Key file {self.key_file} not found")
        with open(self.key_file, 'rb') as f:
            self.key = f.read()
        self.cipher = Fernet(self.key)
        return self.key
    
    def encrypt_file(self, input_file, output_file=None):
        if not self.cipher:
            self.load_key()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        if output_file is None:
            output_file = input_file + '.encrypted'
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted_data)
        
        print(f"File encrypted: {output_file}")
        return output_file
    
    def decrypt_file(self, input_file, output_file=None):
        if not self.cipher:
            self.load_key()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        if output_file is None:
            if input_file.endswith('.encrypted'):
                output_file = input_file[:-10]
            else:
                output_file = input_file + '.decrypted'
        
        with open(input_file, 'rb') as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        
        print(f"File decrypted: {output_file}")
        return output_file

def main():
    encryptor = FileEncryptor()
    
    action = input("Choose action (generate_key/encrypt/decrypt): ").strip().lower()
    
    if action == 'generate_key':
        encryptor.generate_key()
    elif action == 'encrypt':
        input_file = input("Enter file to encrypt: ").strip()
        encryptor.encrypt_file(input_file)
    elif action == 'decrypt':
        input_file = input("Enter file to decrypt: ").strip()
        encryptor.decrypt_file(input_file)
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()import os
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

def generate_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_file_integrity(original_path, decrypted_path):
    return generate_checksum(original_path) == generate_checksum(decrypted_path)