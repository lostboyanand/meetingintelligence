# Meeting Intelligence Platform

This platform enables users to upload meeting recordings and extract actionable insights using AI. The system transcribes the audio, identifies action items, decisions, and key topics, and allows users to query the meeting content using natural language.

## Features

- Audio/video file upload and processing
- Automatic transcription using Whisper
- AI-powered extraction of:
  - Meeting summaries
  - Action items and assignees
  - Key decisions
  - Important topics
- Interactive Q&A about meeting content
- Meeting dashboard and analytics

## Technical Stack

- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Backend**: Python FastAPI
- **Database**: SQLite
- **AI Components**:
  - OpenAI Whisper (transcription)
  - Amazon Bedrock (LLM for analysis)
  - Chroma (vector search)

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js and npm (for frontend development)
- AWS account with Bedrock access
- AWS credentials configured with proper permissions

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd meeting-intelligence-platform
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**

   Ensure your AWS credentials are configured either via environment variables, AWS CLI, or credentials file:
   ```
   aws configure
   ```
   The user/role needs permissions for Amazon Bedrock services.

5. **Environment variables**

   Create a `.env` file in the project root with the following variables:
   ```
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

6. **Initialize the database**
   ```bash
   # The database will be initialized automatically on first run
   ```

7. **Start the backend server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

The frontend is a single HTML file that can be served by any web server:

1. **Open `index.html` in your web browser**
   - You can serve it using Python's built-in HTTP server:
     ```bash
     python -m http.server 3000
     ```
   - Then open http://localhost:3000 in your browser

2. **Configure API endpoint**
   
   If you're not running the backend on the default URL, update the API_BASE variable in the `index.html` file:
   ```javascript
   const API_BASE = 'http://localhost:8000/api';
   ```

## Usage

1. **Upload a meeting recording**
   - Enter your email address
   - Choose an audio or video file
   - Add an optional meeting title
   - Click "Upload Meeting"

2. **Browse meetings**
   - Load user emails
   - Select a specific user
   - Load meetings for that user
   - Select a meeting to view details

3. **View meeting insights**
   - See summary, key topics, action items and decisions
   - Review transcription details (toggle visibility)

4. **Ask questions about the meeting**
   - With a meeting selected, enter a question in the Q&A section
   - Click "Ask About This Meeting" to get AI-generated answers based on the meeting content

## Required Configurations

### Amazon Bedrock Models

The system uses the following Amazon Bedrock models:
- `amazon.titan-text-premier-v1:0` for analysis and question answering
- `amazon.titan-embed-text-v2:0` for embeddings

Ensure your AWS account has access to these models.

### Storage Requirements

- Local storage for uploaded audio/video files
- SQLite database for structured data
- ChromaDB for vector embeddings

### Hardware Recommendations

- At least 4GB RAM for running Whisper models
- Sufficient disk space for storing audio/video files and database

## Troubleshooting

- **Upload failures**: Check file size and format compatibility
- **Transcription errors**: Ensure audio quality is clear; try different file formats
- **AWS connectivity issues**: Verify AWS credentials and permissions
- **Missing dependencies**: Run `pip install -r requirements.txt` again
- **Database errors**: Delete the `meetings.db` file to reset and let it reinitialize

## Security Notes

This implementation focuses on functionality rather than security. For production use, implement:
- Proper user authentication
- API rate limiting
- Secure file handling
- HTTPS encryption
- Environment variable protection

## License
The source code is totally owned by the owner sumit anand Please take proper permission for use.