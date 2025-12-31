
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes) -> tuple:
        key = PBKDF2(self.password, salt, dkLen=32, count=1000000)
        return key[:16], key[16:]

    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()

            salt = get_random_bytes(self.salt_length)
            key, iv = self.derive_key(salt)

            cipher = AES.new(key, AES.MODE_CBC, iv)
            ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

            with open(output_path, 'wb') as f:
                f.write(salt + ciphertext)

            return True
        except Exception:
            return False

    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            salt = data[:self.salt_length]
            ciphertext = data[self.salt_length:]

            key, iv = self.derive_key(salt)
            cipher = AES.new(key, AES.MODE_CBC, iv)
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

            with open(output_path, 'wb') as f:
                f.write(plaintext)

            return True
        except Exception:
            return False

    def calculate_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_file = "test_document.txt"
    encrypted_file = "encrypted.dat"
    decrypted_file = "decrypted.txt"

    with open(test_file, 'w') as f:
        f.write("This is a secret document containing sensitive information.")

    if encryptor.encrypt_file(test_file, encrypted_file):
        print(f"File encrypted successfully: {encrypted_file}")
        print(f"Encrypted file hash: {encryptor.calculate_hash(encrypted_file)}")

    if encryptor.decrypt_file(encrypted_file, decrypted_file):
        print(f"File decrypted successfully: {decrypted_file}")
        print(f"Decrypted file hash: {encryptor.calculate_hash(decrypted_file)}")

    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)

if __name__ == "__main__":
    main()