from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import os
import uuid
from datetime import datetime
from typing import Optional
import sqlite3
import subprocess  # For extracting audio from video
import json
import whisper
from langchain_community.llms import Bedrock
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings, BedrockLLM  # <-- Use this for embeddings
from dotenv import load_dotenv
from ffmpeg_setup import FFMPEG_PATH
# from aws_config import get_aws_session

# # Get AWS session
# aws_session = get_aws_session()
# Initialize Whisper model
load_dotenv()
whisper_model = whisper.load_model("base")

# LangChain Bedrock and Chroma setup
embedding = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)
vectorstore = Chroma(collection_name="meeting_insights", embedding_function=embedding)

router = APIRouter()

@router.get("/users/")
async def list_users():
    """
    List all users (emails) who have recorded meetings
    """
    conn = sqlite3.connect("meetings.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all unique emails that have meetings
    cursor.execute(
        """SELECT DISTINCT user_email 
           FROM meetings
           ORDER BY user_email"""
    )
    
    users = [row["user_email"] for row in cursor.fetchall()]
    conn.close()
    
    return {"users": users}

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
    Automatically registers new users if email not found.
    """
    # First check/register the user in one step
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if not user:
        # Register new user automatically
        cursor.execute(
            "INSERT INTO users (email, created_at) VALUES (?, ?)",
            (email, datetime.now().isoformat())
        )
        conn.commit()
    
    conn.close()
    
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
                    FFMPEG_PATH, '-i', file_path, '-q:a', '0', '-map', 'a', audio_path
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
        # Updated to pass user_email to the vector store function
        store_in_vector_db(meeting_id, transcription, segments, summary, key_topics, user_email)
        
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
    
    model_id = "amazon.titan-text-premier-v1:0"
    # Use an Amazon Bedrock model
    
    try:
        llm = BedrockLLM(
            model_id=model_id,
            region_name="us-east-1"
        )
        result_text = llm.invoke(prompt)
        
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

def store_in_vector_db(meeting_id, transcription, segments, summary, key_topics, user_email):
    """
    Store meeting information in vector database for semantic search using LangChain's Chroma
    Include user_email in metadata for better filtering
    """
    try:
        # For better chunking, split the transcription into paragraph-sized chunks
        # so we can retrieve more relevant context around matches
        chunk_size = 500  # characters per chunk
        overlap = 100     # overlap between chunks
        
        if transcription:
            transcription_chunks = []
            for i in range(0, len(transcription), chunk_size - overlap):
                chunk = transcription[i:i + chunk_size]
                transcription_chunks.append(chunk)
            
            # Store each chunk of the transcript
            for i, chunk in enumerate(transcription_chunks):
                metadata = {
                    "meeting_id": meeting_id,
                    "type": "transcript_chunk",
                    "chunk_index": i,
                    "total_chunks": len(transcription_chunks),
                    "user_email": user_email
                }
                vectorstore.add_texts(
                    texts=[chunk],
                    metadatas=[metadata]
                )
        
        # Store summary separately
        if summary:
            summary_metadata = {
                "meeting_id": meeting_id,
                "type": "summary",
                "user_email": user_email
            }
            vectorstore.add_texts(
                texts=[summary],
                metadatas=[summary_metadata]
            )
        
        # Store topics for better retrieval
        if key_topics:
            topics_text = "Key topics discussed: " + ", ".join(key_topics)
            topics_metadata = {
                "meeting_id": meeting_id,
                "type": "key_topics",
                "user_email": user_email
            }
            vectorstore.add_texts(
                texts=[topics_text],
                metadatas=[topics_metadata]
            )
        
        # Store segments if available
        if segments:
            for i, segment in enumerate(segments):
                segment_text = segment.get("text", "")
                if segment_text:
                    metadata = {
                        "meeting_id": meeting_id,
                        "type": "segment",
                        "start_time": segment.get("start", 0),
                        "end_time": segment.get("end", 0),
                        "segment_index": i,
                        "user_email": user_email
                    }
                    vectorstore.add_texts(
                        texts=[segment_text],
                        metadatas=[metadata]
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
async def list_meetings(email: Optional[str] = None):
    """
    List all meetings for a specific user by email.
    If no email is provided, returns all meetings.
    """
    conn = sqlite3.connect("meetings.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if email:
        # Filter meetings by email
        cursor.execute(
            """SELECT id, title, user_email, status, created_at, completed_at 
               FROM meetings 
               WHERE user_email = ? 
               ORDER BY created_at DESC""",
            (email,)
        )
    else:
        # Return all meetings if no email specified
        cursor.execute(
            """SELECT id, title, user_email, status, created_at, completed_at 
               FROM meetings 
               ORDER BY created_at DESC"""
        )
    
    meetings = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"email": email, "meetings": meetings, "count": len(meetings)}


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

@router.get("/ask/")
async def ask_question(question: str, meeting_id: str):
    """
    Ask a question about a specific meeting and get an AI-generated answer.
    Uses RAG (Retrieval Augmented Generation) to provide relevant answers.
    """
    try:
        # Check if meeting exists first
        conn = sqlite3.connect("meetings.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT title, summary, key_topics FROM meetings WHERE id = ?", (meeting_id,))
        meeting_data = cursor.fetchone()
        
        if not meeting_data:
            conn.close()
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        meeting_info = dict(meeting_data)
        
        # Parse key topics if available
        if meeting_info.get("key_topics"):
            try:
                meeting_info["key_topics"] = json.loads(meeting_info["key_topics"])
            except:
                meeting_info["key_topics"] = []
        
        # Get additional meeting data based on question
        cursor.execute("SELECT description FROM decisions WHERE meeting_id = ?", (meeting_id,))
        decisions = [row["description"] for row in cursor.fetchall()]
        
        cursor.execute("SELECT description, assignee FROM action_items WHERE meeting_id = ?", (meeting_id,))
        action_items = [f"{row['description']} (Assigned to: {row['assignee']})" for row in cursor.fetchall()]
        
        conn.close()
        
        # Get relevant content from vector search
        filter_dict = {"meeting_id": meeting_id}
        
        # Get more context (10 chunks) to ensure we have enough information
        search_results = vectorstore.similarity_search(
            query=question,
            k=10,
            filter=filter_dict
        )
        
        # Format the context
        contexts = []
        
        # First add the meeting metadata
        contexts.append(f"Meeting title: {meeting_info['title']}")
        contexts.append(f"Meeting summary: {meeting_info.get('summary', '')}")
        
        # Add key topics if available
        if meeting_info.get("key_topics"):
            contexts.append(f"Key topics: {', '.join(meeting_info['key_topics'])}")
        
        # Add decisions if available
        if decisions:
            contexts.append("Key decisions:")
            for decision in decisions:
                contexts.append(f"- {decision}")
        
        # Add action items if available
        if action_items:
            contexts.append("Action items:")
            for item in action_items:
                contexts.append(f"- {item}")
        
        # Add content from vector search
        for doc in search_results:
            if doc.metadata.get("type") == "transcript_chunk":
                contexts.append(f"Transcript content: {doc.page_content}")
            elif doc.metadata.get("type") == "segment":
                contexts.append(f"Transcript segment (Time {doc.metadata.get('start_time')} to {doc.metadata.get('end_time')}): {doc.page_content}")
            else:
                contexts.append(f"{doc.metadata.get('type', 'Content')}: {doc.page_content}")
        
        # Create the prompt
        context_text = "\n\n".join(contexts)
        
        prompt = f"""
            You are an experienced AI research assistant analyzing meeting transcripts. Your task is to provide highly detailed, factual, and comprehensive answers to questions about meeting "{meeting_info['title']}".

            USER QUESTION: "{question}"

            INSTRUCTIONS:
            1. Answer ONLY based on the CONTEXT provided below.
            2. Include ALL relevant details from the context that address the question.
            3. Structure your answer with clear paragraphs and bullet points when appropriate.
            4. If answering about meeting summary, include ALL key points, topics, decisions, and action items mentioned.
            5. For questions about specific topics, provide extensive details including who discussed them and what was concluded.
            6. If discussing action items, include ALL details about assignees, deadlines, and specific responsibilities.
            7. When mentioning decisions, explain the full context of how they were reached.
            8. For questions about participants, detail everyone's contributions and roles.
            9. If the EXACT information requested is NOT in the context, state clearly: "Based on the meeting information provided, I cannot find specific details about [topic]. The available context covers [what IS available]."
            10. Maintain a professional, analytical tone throughout your response.
            11. DO NOT invent, assume, or infer information not explicitly stated in the context.
            12. DO NOT reference sources or information outside the provided context.
            13. PRIORITIZE accuracy over brevity - your answer should be comprehensive.

            CONTEXT:
            {context_text}

            Begin your answer now, ensuring it is thorough, well-structured, and directly addresses the question using only information from the context provided.
            """
        
        # Generate answer
        llm = BedrockLLM(
            model_id="amazon.titan-text-premier-v1:0",
            region_name="us-east-1"
        )
        
        answer = llm.invoke(prompt)
        
        # Return the answer
        return {
            "question": question,
            "answer": answer,
            "meeting_id": meeting_id,
            "meeting_title": meeting_info["title"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
