# medasktrial_V_1




## Project Title: MedAskTrial - Document Chat Application

### Description

**MedAskTrial** is a **Document Chat Application** built using **Django**, **FAISS**, **local embedding models**, and **PostgreSQL**. The application allows users to upload medical documents and interact with them via a chat interface. It uses **Retrieval-Augmented Generation (RAG)** to answer user queries based on document content. The project supports storing user conversations and sessions, managing multiple documents, and exporting conversations to various file formats. **NLTK** is used for text preprocessing and cleaning.

### Features

- Document upload and chunking
- Embeddings generation using a local model (e.g., `paraphrase-MiniLM-L6-v2`)
- FAISS for similarity search
- Dynamic document and session handling
- Conversation and session management in PostgreSQL
- Exporting conversations to DOCX, JSON, and PPTX formats
- Watermarking exported files
- NLTK for text preprocessing and tokenization
- Contextual chat interface using OpenAI's GPT models

### Requirements

- Python 3.8.10 (due to compatibility with FAISS CPU)
- Django 4.2.13
- PostgreSQL
- OpenAI API key for chat-based responses
- NLTK
- AWS Polly Keys For text-to-speech functionality



### Installation

1. **Clone the repository**:
  
   git clone 
   cd medasktrial


2. **Create and activate a virtual environment**:
   
   python3.8 -m venv venv
   source venv/bin/activate
   for windows:
   venv\Scripts\activate


3. **Install dependencies**:
  
   pip install -r requirements.txt
  

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add the following keys:
   
   OPENAI_API_KEY=your_openai_api_key
  
   AWS_ACCESS_KEY_ID='aws polly id'
   AWS_SECRET_ACCESS_KEY='aws secret access key'
   

5. **Run database migrations**:
   
   python manage.py migrate

   python manage.py makemigrations
 

6. **Install NLTK stopwords and other resources**:
   Run the following commands to download necessary NLTK data:
   
   python manage.py shell

   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
  

7. **Update NLTK data path (if necessary)**:
   In `settings.py`, add the following lines if you need to specify a custom NLTK data path:

   import nltk

   # Append NLTK data path
   nltk.data.path.append('C:\\Users\\user\\nltk_data')  # Update this path as needed
 

8. **Create a superuser**:
  
   python manage.py createsuperuser
9. **Collect the static files** :
   python manage.py collectstatic

10. **Start the development server**:
  
   python manage.py runserver
  

### Usage

1. **Visit the Application**:
   - Access the application at [MedAskTrial](https://medasktrial.cardiovalens.com/).

2. **Upload documents**:
   - Navigate to the document upload page in Admin pannel and upload a medical document. The document will be processed, chunked, and embedded using a local embedding model and stored in PostgreSQL.

3. **Interact with the document**:
   - Navigate to the user dashboard and Start a new conversation by asking questions related to the document content. The chat interface will retrieve the most relevant sections of the document and generate responses based on the retrieved content.

4. **Manage sessions and conversations**:
   - Conversations and session data are stored in the PostgreSQL database. Users can return to previous conversations or start new ones  different documents.

5. **Export conversations**:
   - Users can export conversations to DOCX, JSON, or PPTX formats with optional watermarking.

### File Structure


### File Structure


tellmepdf/
│
├── app/                  # App for chat interface and document handling
│   ├── migrations/       # Django migrations
│   ├── models.py         # Models for documents, sessions, and conversation data
│   ├── views.py          # Views for handling chat interactions and document uploads
│   ├── forms.py 
│   ├── urls.py
│   └──  ...              # Other files
├── templates/            # HTML templates for the chat interface
│   ├── admin/       
│   ├── app/             
│   └── user/  
├── static/                # Static files (CSS, JS)
│   ├── css/       
│   ├── js/       
│   ├── media/        
│   └── plugins/             
├── media/  
├── tellmepdf/
│   ├── settings.py/       # Django migrations
│   ├── asgi.py           
│   ├── urls.py            # Django urls
│   └── wsgi  
├── manage.py             # Django management script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── README.md             # Project documentation
└── ...                   # Other files and directories

### Technologies

- **Django**: The web framework used to build the backend.
- **Local Embedding Model**: Used for generating embeddings from text for similarity searches (e.g., `paraphrase-MiniLM-L6-v2`).
- **FAISS**: For similarity search within document chunks.
- **PostgreSQL**: For storing user and session data.
- **NLTK**: For text preprocessing and tokenization (stopword removal, stemming, and lemmatization).
- **OpenAI GPT-4**: Used for generating responses based on document content.

### Future Improvements

- Improve chat response accuracy and contextual understanding.
- Add support for additional file types and processing methods.
- Implement more advanced session management and recovery features.


