from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import os
import uuid
from datetime import datetime
from typing import Optional
import sqlite3
import subprocess  # For extracting audio from video
import json
import whisper
from langchain.llms import Bedrock
from langchain.vectorstores import Chroma
from langchain.embeddings import BedrockEmbeddings

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# LangChain Bedrock and Chroma setup
embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")
vectorstore = Chroma(collection_name="meeting_insights", embedding_function=embedding)

router = APIRouter()

@router.post("/upload-meeting/")
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    email: str = Form(...),
    meeting_title: Optional[str] = Form(None),
):
    """
    Upload a meeting recording (audio or video) with user identifier.
    For video files, the audio will be extracted before processing.
    """
    # Generate a unique meeting ID
    meeting_id = str(uuid.uuid4())
    
    # Create directories if they don't exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed_audio", exist_ok=True)
    
    # Save the uploaded file
    file_path = f"uploads/{meeting_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Determine if it's an audio or video file
    file_extension = os.path.splitext(file.filename)[1].lower()
    audio_path = file_path
    
    # For video files, extract audio
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        audio_path = f"processed_audio/{meeting_id}_audio.wav"
        try:
            # Use ffmpeg to extract audio
            subprocess.run([
                'ffmpeg', '-i', file_path, '-q:a', '0', '-map', 'a', audio_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract audio: {str(e)}")
    
    # Store initial meeting info in database
    meeting_title = meeting_title or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO meetings (id, title, user_email, original_file, audio_file, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            meeting_id, 
            meeting_title, 
            email, 
            file_path, 
            audio_path, 
            "processing", 
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()
    
    # Start background processing
    background_tasks.add_task(process_meeting_audio, meeting_id, audio_path, email)
    
    return {
        "meeting_id": meeting_id,
        "status": "processing",
        "message": "Meeting upload successful. Processing has started."
    }


async def process_meeting_audio(meeting_id: str, audio_path: str, user_email: str):
    """
    Process the meeting audio file:
    1. Transcribe with Whisper
    2. Extract insights with Amazon Bedrock
    3. Update database with results
    4. Prepare data for search functionality
    """
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    
    try:
        # Update status to transcribing
        cursor.execute(
            "UPDATE meetings SET status = ? WHERE id = ?", 
            ("transcribing", meeting_id)
        )
        conn.commit()
        
        # Step 1: Transcribe with Whisper
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        segments = result.get("segments", [])
        
        # Update status and save transcription
        cursor.execute(
            "UPDATE meetings SET transcription = ?, status = ? WHERE id = ?", 
            (transcription, "analyzing", meeting_id)
        )
        conn.commit()
        
        # Step 2: Extract insights with Amazon Bedrock
        analysis_result = analyze_transcript(transcription)
        summary = analysis_result.get("summary", "")
        action_items = analysis_result.get("action_items", [])
        key_topics = analysis_result.get("key_topics", [])
        decisions = analysis_result.get("decisions", [])
        
        # Step 3: Store results in database
        now = datetime.now().isoformat()
        
        # Update meeting with summary and analysis results
        cursor.execute(
            """UPDATE meetings SET 
                summary = ?, 
                key_topics = ?, 
                status = ?, 
                completed_at = ? 
               WHERE id = ?""", 
            (summary, json.dumps(key_topics), "completed", now, meeting_id)
        )
        
        # Insert action items
        for item in action_items:
            item_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO action_items 
                   (id, meeting_id, description, assignee, due_date, status, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    item_id, 
                    meeting_id, 
                    item.get("description", ""), 
                    item.get("assignee", "Unassigned"), 
                    item.get("due_date", None),
                    "pending", 
                    now
                )
            )
        
        # Insert decisions
        for decision in decisions:
            decision_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO decisions
                   (id, meeting_id, description, created_at)
                   VALUES (?, ?, ?, ?)""",
                (decision_id, meeting_id, decision, now)
            )
        
        # Step 4: Prepare data for search functionality
        store_in_vector_db(meeting_id, transcription, segments, summary, key_topics)
        
        conn.commit()
        
    except Exception as e:
        # In case of error, update status
        cursor.execute(
            "UPDATE meetings SET status = ? WHERE id = ?", 
            (f"error: {str(e)}", meeting_id)
        )
        conn.commit()
        print(f"Error processing meeting {meeting_id}: {str(e)}")
        
    finally:
        conn.close()


def analyze_transcript(transcription: str):
    """
    Use Amazon Bedrock to analyze the transcript and extract insights
    """
    prompt = f"""
    Analyze this meeting transcript and provide the following information in JSON format:
    
    Meeting Transcript:
    {transcription}
    
    Please extract:
    1. A concise summary (max 250 words)
    2. Action items with assignees (if mentioned)
    3. Key decisions made during the meeting
    4. Main topics discussed (5-7 topics)
    
    Format your response as a JSON object with the following structure:
    {{
        "summary": "meeting summary text",
        "action_items": [
            {{"description": "action item description", "assignee": "person name", "due_date": "YYYY-MM-DD or null"}}
        ],
        "decisions": [
            "decision 1",
            "decision 2"
        ],
        "key_topics": [
            "topic 1",
            "topic 2"
        ]
    }}
    """
    
    model_id = "anthropic.claude-v2"  # Use an appropriate model from Amazon Bedrock
    
    try:
        llm = Bedrock(
            model_id=model_id,
            region_name="us-east-1"
        )
        result_text = llm(prompt)
        
        result_json = extract_json_from_text(result_text)
        
        # Ensure all expected keys exist
        if not all(k in result_json for k in ["summary", "action_items", "decisions", "key_topics"]):
            print("Warning: Some expected keys missing from LLM response")
            
        return {
            "summary": result_json.get("summary", ""),
            "action_items": result_json.get("action_items", []),
            "decisions": result_json.get("decisions", []),
            "key_topics": result_json.get("key_topics", [])
        }
        
    except Exception as e:
        print(f"Error analyzing transcript: {str(e)}")
        # Return a default structure in case of error
        return {
            "summary": "Error analyzing transcript.",
            "action_items": [],
            "decisions": [],
            "key_topics": []
        }

def extract_json_from_text(text):
    """Helper function to extract JSON from LLM response"""
    try:
        # First attempt: Try to parse the entire text as JSON
        return json.loads(text)
    except:
        try:
            # Second attempt: Look for JSON within the text
            import re
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
    
    # Return empty dict if JSON extraction fails
    print("Failed to extract JSON from LLM response")
    return {}

def store_in_vector_db(meeting_id, transcription, segments, summary, key_topics):
    """
    Store meeting information in vector database for semantic search using LangChain's Chroma
    """
    try:
        metadata = {
            "meeting_id": meeting_id,
            "type": "full_transcript",
            "summary": summary,
            "topics": ", ".join(key_topics)
        }
        vectorstore.add_texts(
            texts=[transcription],
            metadatas=[metadata]
        )
        
        if segments:
            segment_texts = [segment.get("text", "") for segment in segments]
            segment_metadatas = []
            for i, segment in enumerate(segments):
                metadata = {
                    "meeting_id": meeting_id,
                    "type": "segment",
                    "start_time": segment.get("start", 0),
                    "end_time": segment.get("end", 0),
                    "segment_index": i
                }
                segment_metadatas.append(metadata)
            vectorstore.add_texts(
                texts=segment_texts,
                metadatas=segment_metadatas
            )
        
        print(f"Successfully stored meeting {meeting_id} in vector database")
    except Exception as e:
        print(f"Error storing in vector DB: {str(e)}")

@router.post("/verify-email/")
async def verify_email(email: str = Form(...)):
    """
    Verify or register a user by email
    """
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if not user:
        # Register new user
        cursor.execute(
            "INSERT INTO users (email, created_at) VALUES (?, ?)",
            (email, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return {"message": "New user registered", "email": email}
    
    conn.close()
    return {"message": "User verified", "email": email}


@router.get("/meetings/")
async def list_meetings(email: str):
    """
    List all meetings for a specific user by email
    """
    conn = sqlite3.connect("meetings.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT id, title, status, created_at, completed_at 
           FROM meetings 
           WHERE user_email = ? 
           ORDER BY created_at DESC""",
        (email,)
    )
    
    meetings = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"email": email, "meetings": meetings}


@router.get("/meetings/{meeting_id}")
async def get_meeting_details(meeting_id: str):
    """
    Get detailed information about a specific meeting
    """
    conn = sqlite3.connect("meetings.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get meeting data
    cursor.execute(
        """SELECT id, title, user_email, status, transcription, summary, 
                  key_topics, created_at, completed_at 
           FROM meetings 
           WHERE id = ?""", 
        (meeting_id,)
    )
    meeting_row = cursor.fetchone()
    
    if not meeting_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    meeting = dict(meeting_row)
    
    # Parse JSON fields
    if meeting.get("key_topics"):
        try:
            meeting["key_topics"] = json.loads(meeting["key_topics"])
        except:
            meeting["key_topics"] = []
    
    # Get action items
    cursor.execute(
        """SELECT id, description, assignee, due_date, status, created_at 
           FROM action_items 
           WHERE meeting_id = ?""",
        (meeting_id,)
    )
    action_items = [dict(row) for row in cursor.fetchall()]
    meeting["action_items"] = action_items
    
    # Get decisions
    cursor.execute(
        """SELECT id, description, created_at 
           FROM decisions 
           WHERE meeting_id = ?""",
        (meeting_id,)
    )
    decisions = [dict(row) for row in cursor.fetchall()]
    meeting["decisions"] = decisions
    
    conn.close()
    
    return meeting

@router.get("/search/")
async def search_meetings(query: str, email: Optional[str] = None, limit: int = 5):
    """
    Search through meeting content using LangChain vector similarity search
    """
    try:
        # Use LangChain's similarity search with metadata filter if email is provided
        filter_dict = {"user_email": email} if email else None
        
        search_results = vectorstore.similarity_search(
            query=query,
            k=limit,
            filter=filter_dict
        )
        
        # Process and format results
        formatted_results = []
        for doc in search_results:
            # Extract metadata from the document
            metadata = doc.metadata
            meeting_id = metadata.get("meeting_id")
            
            # Get additional meeting info from database for enriching results
            conn = sqlite3.connect("meetings.db")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, title, user_email, created_at, summary
                   FROM meetings 
                   WHERE id = ?""",
                (meeting_id,)
            )
                
            meeting_row = cursor.fetchone()
            conn.close()
            
            if not meeting_row:
                continue  # Skip if meeting not found
                
            meeting_info = dict(meeting_row)
            
            # Format the result with rich context
            result = {
                "meeting_id": meeting_id,
                "meeting_title": meeting_info.get("title"),
                "meeting_date": meeting_info.get("created_at"),
                "meeting_summary": meeting_info.get("summary"),
                "segment_type": metadata.get("type", "unknown"),
                "content": doc.page_content,
                "context": {
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "topics": metadata.get("topics", "")
                },
                "score": getattr(doc, "score", None)  
            }
            
            formatted_results.append(result)
        
        return {"query": query, "results": formatted_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")