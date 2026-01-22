from cryptography.fernet import Fernet
import os

class FileEncryptor:
    def __init__(self, key_path='secret.key'):
        self.key_path = key_path
        self.key = None
        self.cipher = None
        
    def generate_key(self):
        self.key = Fernet.generate_key()
        with open(self.key_path, 'wb') as key_file:
            key_file.write(self.key)
        print(f"Key generated and saved to {self.key_path}")
        return self.key
    
    def load_key(self):
        if not os.path.exists(self.key_path):
            raise FileNotFoundError(f"Key file {self.key_path} not found")
        with open(self.key_path, 'rb') as key_file:
            self.key = key_file.read()
        self.cipher = Fernet(self.key)
        return self.key
    
    def encrypt_file(self, input_path, output_path=None):
        if not self.cipher:
            self.load_key()
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found")
        
        if output_path is None:
            output_path = input_path + '.encrypted'
        
        with open(input_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(output_path, 'wb') as file:
            file.write(encrypted_data)
        
        print(f"File encrypted and saved to {output_path}")
        return output_path
    
    def decrypt_file(self, input_path, output_path=None):
        if not self.cipher:
            self.load_key()
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found")
        
        if output_path is None:
            if input_path.endswith('.encrypted'):
                output_path = input_path[:-10]
            else:
                output_path = input_path + '.decrypted'
        
        with open(input_path, 'rb') as file:
            encrypted_data = file.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)
        
        print(f"File decrypted and saved to {output_path}")
        return output_path

def main():
    encryptor = FileEncryptor()
    
    action = input("Choose action (generate_key/encrypt/decrypt): ").strip().lower()
    
    if action == 'generate_key':
        encryptor.generate_key()
    elif action == 'encrypt':
        input_file = input("Enter input file path: ").strip()
        output_file = input("Enter output file path (optional): ").strip()
        if not output_file:
            output_file = None
        encryptor.encrypt_file(input_file, output_file)
    elif action == 'decrypt':
        input_file = input("Enter input file path: ").strip()
        output_file = input("Enter output file path (optional): ").strip()
        if not output_file:
            output_file = None
        encryptor.decrypt_file(input_file, output_file)
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()