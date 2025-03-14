import asyncio
import os
import shutil
import subprocess
import requests
import time
import base64
import tempfile
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

# Import FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Import Deepgram if you want to keep speech-to-text capability
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions

# Load environment variables
load_dotenv()

app = FastAPI(title="Voice Assistant API", description="API for voice-based AI assistant interactions")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class TextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class AudioRequest(BaseModel):
    session_id: Optional[str] = None

class Response(BaseModel):
    text_response: str
    audio_base64: Optional[str] = None

# Store conversations by session ID
conversations = {}

class LanguageModelProcessor:
    def __init__(self, session_id=None):
        self.session_id = session_id
        self.llm = ChatGroq(temperature=0, model="mixtral-8x7b-32768")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load system prompt from file or use default
        system_prompt = ""
        try:
            with open('system_prompt.txt', 'r') as file:
                system_prompt = file.read().strip()
        except FileNotFoundError:
            system_prompt = "You are a helpful assistant that responds accurately and concisely."

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start_time = time.time()
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"🤖 AI ({elapsed_time}ms): {response['text']}")
        return response['text']

class ElevenLabsTTS:
    """Class to handle Text-to-Speech (TTS) requests using Eleven Labs API with voice selection."""
    
    ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
    MALE_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID")
    FEMALE_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID_2")

    def __init__(self, voice_type="male"):
        """Initialize with default voice type."""
        self.voice_type = voice_type
        
    def get_voice_id(self):
        """Return the appropriate voice ID based on selected voice type."""
        if self.voice_type.lower() == "female":
            return self.FEMALE_VOICE_ID
        else:  # Default to male voice
            return self.MALE_VOICE_ID

    def get_audio(self, text, voice_type=None):
        """Convert text to speech using Eleven Labs API and return the audio data.
        
        Args:
            text (str): The text to convert to speech
            voice_type (str, optional): Override the instance voice type ('male' or 'female')
        """
        # Update voice type if provided
        if voice_type:
            self.voice_type = voice_type
            
        voice_id = self.get_voice_id()
        
        if not self.ELEVEN_LABS_API_KEY:
            raise ValueError("❌ Missing Eleven Labs API Key! Set it in the .env file.")

        if not voice_id:
            raise ValueError(f"❌ Missing Eleven Labs Voice ID for {self.voice_type} voice! Set it in the .env file.")

        ELEVEN_LABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": self.ELEVEN_LABS_API_KEY,
            "Content-Type": "application/json"
        }

        # Modify text to enhance speech clarity and emotion
        text = self.enhance_text_with_emotions(text)

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.2,  # Increase slightly for a balanced natural flow
                "similarity_boost": 0.85  # Enhance voice realism
            }
        }

        print(f"🔄 Sending enhanced TTS request to Eleven Labs ({self.voice_type} voice): {text}")

        try:
            response = requests.post(ELEVEN_LABS_URL, headers=headers, json=payload)
            if response.status_code != 200:
                print(f"❌ Eleven Labs API Error {response.status_code}: {response.text}")
                return None

            print(f"✅ Eleven Labs TTS request successful with {self.voice_type} voice")
            return response.content

        except Exception as e:
            print(f"⚠️ Error during TTS request: {e}")
            return None

    def enhance_text_with_emotions(self, text):
        """Modify text to add emotional expression and better speech clarity."""
        
        # Adjust pauses and emphasis
        text = text.replace("...", ",,")  # Pause  
        text = text.replace("!!", "!")  # Excitement  
        text = text.replace("???", "?")  # Questioning tone  

        # Emotion-based text modifications
        if "hello" in text.lower():
            return "Hey there!! It's great to see you. I'm QuickAgent—your friendly AI assistant. 😊 How can I help you today?"  
        elif "goodbye" in text.lower():
            return "Goodbye, my friend!! It was a pleasure assisting you. See you soon! 👋"  
        elif "what can you do" in text.lower():
            return "I can help you with a variety of tasks—like answering questions, setting reminders, or analyzing data! What do you need help with today?"  
        else:
            return text

class DeepgramTranscriber:
    """Handle audio transcription using Deepgram."""
    
    def __init__(self):
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        
    async def transcribe_audio_file(self, audio_data):
        """Transcribe audio file using Deepgram's API."""
        try:
            # Ensure audio_data is bytes
            if not isinstance(audio_data, bytes):
                print(f"❌ Warning: audio_data is not bytes but {type(audio_data)}")
                if isinstance(audio_data, str):
                    audio_data = audio_data.encode('utf-8')
                else:
                    raise TypeError(f"Expected bytes, got {type(audio_data)}")
            
            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Initialize Deepgram client with basic config
            options = DeepgramClientOptions(options={"keepalive": "true"})
            
            if self.deepgram_api_key is None:
                raise ValueError("Deepgram API key is not set.")
            else:
                deepgram = DeepgramClient(self.deepgram_api_key, options)
            
            # Make direct API call using requests instead of SDK methods
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
                "Content-Type": "audio/wav"
            }
            
            params = {
                "model": "nova-2",
                "punctuate": "true",
                "language": "en-US",
                "smart_format": "true"
            }
            
            with open(temp_file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                response = requests.post(url, headers=headers, params=params, data=audio_bytes)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if response.status_code != 200:
                print(f"❌ Deepgram API error: {response.status_code} - {response.text}")
                return ""
                
            # Parse response JSON
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript
            
        except Exception as e:
            print(f"❌ Error in Deepgram transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
# API endpoints
@app.get("/")
async def root():
    return {"message": "Voice Assistant API is running. See /docs for API documentation."}

@app.post("/process_text", response_model=Response)
async def process_text(request: TextRequest, voice_type: str = Form(None)):
    """Process text input and return AI response with optional TTS audio."""
    session_id = request.session_id or "default"
    
    # Get or create conversation processor for this session
    if session_id not in conversations:
        conversations[session_id] = LanguageModelProcessor(session_id)
    
    processor = conversations[session_id]
    
    # Process the text input
    text_response = processor.process(request.text)
    
    # Generate audio response with specified voice type
    tts = ElevenLabsTTS()
    audio_data = tts.get_audio(text_response, voice_type)
    
    # Return the response
    response = {"text_response": text_response}
    
    if audio_data:
        response["audio_base64"] = base64.b64encode(audio_data).decode('utf-8')
    
    return response

@app.post("/process_audio")
async def process_audio(
    audio: UploadFile = File(...), 
    session_id: Optional[str] = Form(None),
    voice_type: Optional[str] = Form(None)
):
    """Process audio input, transcribe it, and return AI response with optional TTS audio."""
    session_id = session_id or "default"
    
    # Get or create conversation processor for this session
    if session_id not in conversations:
        conversations[session_id] = LanguageModelProcessor(session_id)
    
    processor = conversations[session_id]
    
    # Read the audio file as bytes
    audio_data = await audio.read()
    
    # Ensure audio_data is bytes
    if not isinstance(audio_data, bytes):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid audio format"}
        )
    
    # Transcribe the audio
    transcriber = DeepgramTranscriber()
    transcript = await transcriber.transcribe_audio_file(audio_data)
    
    if not transcript:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not transcribe audio"}
        )
    
    # Process the transcribed text
    text_response = processor.process(transcript)
    
    # Generate audio response with specified voice type
    tts = ElevenLabsTTS()
    audio_response_data = tts.get_audio(text_response, voice_type)
    
    # Return the response
    response = {
        "transcript": transcript,
        "text_response": text_response
    }
    
    if audio_response_data:
        response["audio_base64"] = base64.b64encode(audio_response_data).decode('utf-8')
    
    return response
@app.post("/stream_audio")
async def stream_audio_response(request: TextRequest, voice_type: str = Form(None)):
    """Process text and stream audio response."""
    session_id = request.session_id or "default"
    
    # Get or create conversation processor for this session
    if session_id not in conversations:
        conversations[session_id] = LanguageModelProcessor(session_id)
    
    processor = conversations[session_id]
    
    # Process the text input
    text_response = processor.process(request.text)
    
    # Generate audio response with specified voice type
    tts = ElevenLabsTTS()
    audio_data = tts.get_audio(text_response, voice_type)
    
    if not audio_data:
        return JSONResponse(
            status_code=500,
            content={"error": "Could not generate audio response"}
        )
    
    # Return streaming audio response
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")

@app.delete("/reset_session/{session_id}")
async def reset_session(session_id: str):
    """Reset a conversation session."""
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Session {session_id} has been reset"}
    return {"message": f"Session {session_id} not found"}
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time conversation."""
    await websocket.accept()
    
    session_id = "ws_" + str(time.time())
    processor = LanguageModelProcessor(session_id)
    tts = ElevenLabsTTS()
    transcriber = DeepgramTranscriber()
    
    try:
        while True:
            # Receive audio or text data
            data = await websocket.receive_json()
            
            # Get voice type if provided
            voice_type = data.get("voice_type", None)
            
            if "text" in data:
                # Process text input
                text_input = data["text"]
                text_response = processor.process(text_input)
                
                # Generate audio response
                audio_data = tts.get_audio(text_response, voice_type)
                audio_base64 = base64.b64encode(audio_data).decode('utf-8') if audio_data else None
                
                # Send response
                await websocket.send_json({
                    "text_response": text_response,
                    "audio_base64": audio_base64
                })
                
            elif "audio_base64" in data:
                # Decode audio data - ensure it's bytes
                try:
                    audio_data = base64.b64decode(data["audio_base64"])
                    if not isinstance(audio_data, bytes):
                        raise TypeError("Decoded audio is not bytes")
                except Exception as e:
                    await websocket.send_json({"error": f"Invalid audio data: {str(e)}"})
                    continue
                
                # Transcribe audio
                transcript = await transcriber.transcribe_audio_file(audio_data)
                
                if not transcript:
                    await websocket.send_json({"error": "Could not transcribe audio"})
                    continue
                
                # Process transcript
                text_response = processor.process(transcript)
                
                # Generate audio response
                audio_response_data = tts.get_audio(text_response, voice_type)
                audio_base64 = base64.b64encode(audio_response_data).decode('utf-8') if audio_response_data else None
                
                # Send response
                await websocket.send_json({
                    "transcript": transcript,
                    "text_response": text_response,
                    "audio_base64": audio_base64
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {session_id}")
        if session_id in conversations:
            del conversations[session_id]

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("FASTAPI:app", host="0.0.0.0", port=8000, reload=True)