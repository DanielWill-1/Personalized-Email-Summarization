# ✉️ Personalized Email Summarization
---

An intelligent and secure email summarizer that fetches emails, extracts key content, and presents concise, personalized summaries using NLP and LLMs. Built for productivity, privacy, and powerful insights.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![IMAP](https://img.shields.io/badge/IMAP-email%20access-green?logo=gmail)
![OAuth2](https://img.shields.io/badge/OAuth2-secure%20login-lightgrey?logo=auth0)

---

## 📦 Features

- 🧠 **LLM-Based Summarization**: Extract meaningful insights from lengthy email threads.
- 🔒 **Privacy First**: Local summarization with no email data stored.
- 📬 **Multi-Provider Support**: Connect Gmail (for now)
- 🪄 **Personalization**: Tailored summaries based on your reading preferences.
- 🔐 **OAuth2 Login**: Secure sign-in with Google.

---

Tech Stack
Technology | Purpose
Python | Core programming language
Flask | Web server and backend framework
Socket.IO | Real-time updates for summaries
PyTorch | Running LLMs for summarization
HuggingFace | Transformer models (BART, T5, etc.)
OAuth 2.0 | Secure authentication with Google/Outlook
dotenv | Managing environment variables securely

---

### 🚀 Getting Started

## credentials from GCP

# How to Get Gmail OAuth2 Credentials from GCP

Go to Google Cloud Console:
https://console.cloud.google.com/

Create a New Project (or use an existing one):

Click the project dropdown on top

Select “New Project” → Give it a name → Create

Enable Gmail API:

In the left sidebar, go to APIs & Services > Library

Search for “Gmail API”

Click on it → Press “Enable”

Configure OAuth Consent Screen:

Go to APIs & Services > OAuth consent screen

Choose External if you want to test with your Google account

Fill in App name, support email, and developer contact

Add scopes: Click "Add or Remove Scopes" → add:

```bash

 https://mail.google.com/
 https://www.googleapis.com/auth/gmail.readonly

```bash

Save and continue to publish the app (can stay in "Testing" mode)

Create OAuth Credentials:

Go to APIs & Services > Credentials

Click “Create Credentials” → OAuth client ID

Choose Application Type: Web Application

Add http://localhost:5000/oauth2callback as an Authorized redirect URI

Click Create

Download JSON Credentials:

After creation, click Download JSON from the credentials list

Save this as client_secret.json in the root directory of your project

Add to .gitignore:

bash
Copy
Edit
echo "client_secret.json" >> .gitignore
Your .env should look like:

env
Copy
Edit
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret
REDIRECT_URI=http://localhost:5000/oauth2callback


### 📁 Clone the repo
```bash
git clone https://github.com/DanielWill-1/Personalized-Email-Summarization
cd Personalized-Email-Summarization
python app.py

```
Installation of dependencies
```bash
pip install -r requirements.txt
```bash





---
Summary Logic
Login via OAuth → Get secure email access token

Fetch emails → Use IMAP to retrieve unread/important messages

Run Summarizer → Use LLM (e.g., T5 or BART) for abstract summarization

Display → Return summary with highlights and links
---

Models Used
facebook/bart-large-cnn

t5-small (lightweight alternative)

Supports switching models via config

---

TODO / Roadmap
 Outlook OAuth integration

 User profile & preferences

 Desktop & mobile clients

 Smart tagging and priority detection

 Caching with Database

 Cloud sync and deployment

---

