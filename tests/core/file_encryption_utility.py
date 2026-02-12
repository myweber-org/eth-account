
import os
import sys

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        print(f"File encrypted successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error during encryption: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> [key]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if len(sys.argv) > 4:
        try:
            key = int(sys.argv[4]) % 256
        except ValueError:
            print("Key must be an integer")
            sys.exit(1)
    else:
        key = 42
    
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)
    
    if action == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif action == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Action must be 'encrypt' or 'decrypt'")

if __name__ == "__main__":
    main()