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
    example_usage()