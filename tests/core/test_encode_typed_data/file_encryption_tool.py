import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _xor_operation(self, data: bytes) -> bytes:
        key_bytes = self.key
        key_length = len(key_bytes)
        return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            
            ciphertext = self._xor_operation(plaintext)
            
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            
            print(f"File encrypted successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Encryption failed: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str):
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> [key]")
        print("If key is not provided, it will be prompted securely.")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if len(sys.argv) > 4:
        key = sys.argv[4]
    else:
        import getpass
        key = getpass.getpass("Enter encryption key: ")
    
    if not key:
        print("Error: Key cannot be empty.")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if operation == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
    elif operation == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
    else:
        print(f"Error: Unknown operation '{operation}'. Use 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()