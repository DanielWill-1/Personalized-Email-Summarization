import os
import base64
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from transformers import pipeline, BartTokenizer
from google.auth.transport.requests import Request
import torch

# Set CUDA_LAUNCH_BLOCKING for debugging (optional, can be removed in production)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Initialize BART tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Initialize HuggingFace Summarizer
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer, device=device)

# If modifying these SCOPES, delete the token.json
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def extract_email_body(msg_data):
    """Extracts plain text email content safely from different formats."""
    payload = msg_data['payload']
    body = ""

    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                try:
                    body = base64.urlsafe_b64decode(part['body']['data']).decode()
                    break
                except Exception:
                    continue
    elif 'body' in payload and 'data' in payload['body']:
        try:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode()
        except Exception:
            pass

    return body.strip()

def fetch_emails(service, max_emails=5):
    result = service.users().messages().list(userId='me', maxResults=max_emails).execute()
    messages = result.get('messages', [])
    email_bodies = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        body = extract_email_body(msg_data)
        if body:
            email_bodies.append(body)
    return email_bodies

def summarize_text(text):
    text = text.strip()
    word_count = len(text.split())
    print(f"Text length: {word_count} words")

    if not text:
        return "No content to summarize."
    elif word_count < 30:
        return text  # Return the full message if it's too short

    print("Summarizing now...")
    try:
        # Tokenize and truncate input to max 1024 tokens
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        input_token_count = inputs.input_ids.shape[1]
        print(f"Input token count: {input_token_count}")

        # Adjust max_length and min_length based on input length
        max_len = min(max(50, word_count // 2), 150)  # Reasonable summary length
        min_len = min(25, max_len // 2)

        # Run summarization
        summary = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        return summary[0].get('summary_text', 'Summary generation failed.')
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Fallback to CPU if CUDA fails
        if device == 0:
            print("Retrying summarization on CPU...")
            try:
                cpu_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer, device=-1)
                summary = cpu_summarizer(
                    text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                return summary[0].get('summary_text', 'Summary generation failed.')
            except Exception as cpu_e:
                return f"CPU summarization failed: {cpu_e}"
        return f"Summarization failed: {e}"

def main():
    print(f"Device set to use: {'cuda:0' if device == 0 else 'cpu'}")
    service = authenticate_gmail()
    print("Fetching emails...")
    emails = fetch_emails(service, max_emails=3)

    for i, email in enumerate(emails, start=1):
        print(f"\n📩 Email {i}:\n{email}\n")
        summary = summarize_text(email)
        print(f"📝 Summary {i}:\n{summary}")
        print("\n" + "="*70)

if __name__ == '__main__':
    main()