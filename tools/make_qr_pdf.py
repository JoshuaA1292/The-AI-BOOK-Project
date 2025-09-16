from transformers import AutoTokenizer
token = "[REDACTED_HF_TOKEN]"  # Replace with your token
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=token)
    print("Token works!")
except Exception as e:
    print(f"Token failed: {str(e)}")