import os
import base64
import json
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
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# If modifying these SCOPES, delete the token.json
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Cache file
CACHE_FILE = "summaries_cache.json"

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
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        body = extract_email_body(msg_data)
        if body:
            emails.append({"id": msg['id'], "body": body})
    return emails

def categorize_email(text):
    """Categorize email as Promo, Policies, Personal, or Spam."""
    text = text[:512].lower()  # Limit input size
    # Keyword-based rules for categorization
    promo_keywords = ['sale', 'discount', 'offer', 'deal', 'promotion']
    policy_keywords = ['terms', 'policy', 'guidelines', 'update', 'compliance']
    spam_keywords = ['win', 'free', 'click here', 'unsubscribe', 'limited time offer']
    
    if any(keyword in text for keyword in spam_keywords):
        return "Spam"
    elif any(keyword in text for keyword in policy_keywords):
        return "Policies"
    elif any(keyword in text for keyword in promo_keywords):
        return "Promo"
    
    # Use classifier for sentiment to help with Personal category
    result = classifier(text, truncation=True)
    sentiment = result[0]['label']  # POSITIVE or NEGATIVE
    if sentiment == 'POSITIVE' and not any(keyword in text for keyword in (promo_keywords + policy_keywords)):
        return "Personal"
    return "Personal"  # Default to Personal if unclear

def summarize_text(text, summary_length="medium"):
    text = text.strip()
    word_count = len(text.split())
    print(f"Text length: {word_count} words")

    if not text:
        return "No content to summarize."
    elif word_count < 30:
        return text

    print("Summarizing now...")
    try:
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        input_token_count = inputs.input_ids.shape[1]
        print(f"Input token count: {input_token_count}")

        # Adjust lengths based on user choice
        length_map = {
            "short": {"max_len": 50, "min_len": 15},
            "medium": {"max_len": 100, "min_len": 25},
            "long": {"max_len": 150, "min_len": 50}
        }
        lengths = length_map.get(summary_length.lower(), length_map["medium"])
        max_len = lengths["max_len"]
        min_len = lengths["min_len"]

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

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def main():
    print(f"Device set to use: {'cuda:0' if device == 0 else 'cpu'}")
    
    # Get user inputs
    try:
        max_emails = int(input("How many emails to fetch (1-10, default 3)? ") or 3)
        max_emails = max(1, min(10, max_emails))  # Clamp between 1 and 10
    except ValueError:
        max_emails = 3
        print("Invalid input, using default (3 emails).")
    
    summary_length = input("Summary length (short, medium, long, default medium)? ").lower()
    if summary_length not in ["short", "medium", "long"]:
        summary_length = "medium"
        print("Invalid input, using default (medium).")

    # Load cache
    cache = load_cache()

    # Authenticate and fetch emails
    service = authenticate_gmail()
    print("Fetching emails...")
    emails = fetch_emails(service, max_emails=max_emails)

    for i, email_data in enumerate(emails, start=1):
        email_id = email_data["id"]
        email_body = email_data["body"]
        
        # Check cache
        if email_id in cache:
            summary = cache[email_id]["summary"]
            category = cache[email_id]["category"]
            print(f"\n📩 Email {i} (Cached):")
        else:
            # Categorize and summarize
            category = categorize_email(email_body)
            summary = summarize_text(email_body, summary_length=summary_length)
            # Save to cache
            cache[email_id] = {"summary": summary, "category": category}
            print(f"\n📩 Email {i}:")
        
        print(f"Category: {category}")
        print(f"Content:\n{email_body}\n")
        print(f"📝 Summary {i}:\n{summary}")
        print("\n" + "="*70)

    # Save updated cache
    save_cache(cache)
    print(f"Summaries cached to {CACHE_FILE}")

if __name__ == '__main__':
    main()