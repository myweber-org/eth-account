
import sys
import os

def xor_cipher(data, key):
    """Apply XOR cipher to data with given key."""
    key_length = len(key)
    return bytes([data[i] ^ key[i % key_length] for i in range(len(data))])

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = xor_cipher(plaintext, key)
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        print(f"File encrypted successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error during encryption: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using XOR cipher (same as encrypt)."""
    return encrypt_file(input_path, output_path, key)  # XOR is symmetric

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = sys.argv[4].encode('utf-8')

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if operation == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Invalid operation. Use 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()