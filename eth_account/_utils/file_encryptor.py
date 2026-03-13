import os
import sys

def xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def process_file(input_path: str, output_path: str, key: str):
    """Process file with XOR encryption/decryption."""
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        key_bytes = key.encode('utf-8')
        processed_data = xor_encrypt_decrypt(file_data, key_bytes)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryptor.py <input_file> <output_file> <key>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    if process_file(input_file, output_file, key):
        print("Operation completed.")
    else:
        print("Operation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()