🎙️ Forte Voice – In-App Voice Assistant for ForteBank

Forte Voice is an intelligent, conversational voice assistant built directly into the ForteBank mobile app.
It uses ASR, NLP/NLU, TTS, and a secure backend integration to allow customers to perform banking tasks simply by speaking.

🚀 Features
🎤 Voice Interaction

Forte Voice supports three languages:

Kazakh

Russian

English

Users can speak freely and naturally in any of these languages:

“What’s my card balance?”

“Transfer 2000 tenge”

“Show my expenses”

“How do I pay my credit?”

“Where is the nearest ATM?”

The assistant understands conversational, non-scripted, multilingual speech.

🧠 Intent Understanding (NLP/NLU)

Understands user intent behind phrases and performs tasks such as:

Balance checks

Transaction history

Navigation suggestions

Product information

General FAQs

Assistance flows

🔊 Text-to-Speech (TTS)

Generates natural, human-like voice responses for smooth multilingual dialogue.

🔌 Secure Integration with ForteBank Backend

Through the API Gateway:

retrieves client information

interacts with banking systems

executes informational queries

logs and manages session flow

🏗 Architecture Overview
User Voice
    │
    ▼
[  ASR  ]  Speech Recognition
    │
    ▼
[ NLP/NLU ]  Intent Understanding
    │
    ▼
[ API Gateway ]  Secure Backend Access
    │
    ▼
ForteBank Backend
    │
    ▼
[  TTS  ]  Speech Output
