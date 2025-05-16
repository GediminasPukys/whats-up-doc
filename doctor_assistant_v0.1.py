"""
Real-time Clinical Information Collector

This application:
1. Captures audio continuously using PyAudio
2. Processes audio chunks with OpenAI's Whisper API
3. Extracts clinical information incrementally
4. Saves structured data to a file in real-time
5. Provides growing context to the AI for better understanding

Requirements:
- Python 3.8+
- OpenAI API key
- PyAudio
- NumPy
"""

import os
import json
import queue
import tempfile
import threading
import time
import wave
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pyaudio
import openai

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration parameters
LANGUAGE = os.getenv("LANGUAGE", "LT")  # Default language is Lithuanian
SUPPORTED_LANGUAGES = {
    "LT": "Lithuanian",
    "EN": "English",
    "RU": "Russian"
}

# File for storing collected data (continuously updated)
CLINICAL_DATA_FILE = "clinical_data.json"

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz
CHUNK_SIZE = 1024
RECORD_SECONDS = 3  # Process in 3-second chunks for near real-time experience

# Global variables
audio_queue = queue.Queue()
stop_event = threading.Event()
clinical_data = {}  # Storage for collected clinical information
conversation_history = []  # Store all transcriptions for context

# Initialize clinical data structure
def initialize_clinical_data():
    """Initialize the clinical data structure with empty fields"""
    return {
        "arrival_data": {
            "arrival_datetime": "",
            "arrived_with_referral": None,
            "ambulance": {
                "ambulance_institution": "",
                "ambulance_institution_number": "",
                "ambulance_diagnosis": ""
            },
            "assistance_type": "",
            "consultation_type": "",
            "service_method": ""
        },
        "referral_data": {
            "referring_institution": "",
            "referring_doctor": "",
            "referral_diagnosis": ""
        },
        "medical_data": {
            "complaints": "",
            "medical_history": "",
            "health_assessment": {
                "systolic_blood_pressure": "",
                "diastolic_blood_pressure": "",
                "pulse": "",
                "height": "",
                "weight": "",
                "bmi": "",
                "other_assessment_info": ""
            },
            "dental_formula": ""
        },
        "diagnostics": "",
        "diagnoses": "",
        "allergies": "",
        "health_risk_factors": "",
        "treatment": {
            "medication": "",
            "non_medication": "",
            "recommendations": ""
        },
        "notes": "",
        "newborn_inspection_data": "",
        "medical_certificates": "",
        "notifications": {
            "notification_unable_to_drive": None,
            "driving_prohibition_date": "",
            "notification_unable_to_use_weapon": None
        },
        "sick_leave": {
            "certificate_type": "",
            "certificate_number": "",
            "sick_leave_start_date": "",
            "sick_leave_end_date": "",
            "description": ""
        },
        "interventional_procedures": {
            "procedure_date": "",
            "procedure_name": "",
            "procedure_code": "",
            "description": ""
        }
    }

# Load existing data if available, otherwise initialize
def load_or_initialize_data():
    """Load existing clinical data from file or initialize if not available"""
    if os.path.exists(CLINICAL_DATA_FILE):
        try:
            with open(CLINICAL_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            print("Error loading existing data, initializing new data structure")

    return initialize_clinical_data()

# Function to save data to file
def save_clinical_data():
    """Save the clinical data to file"""
    with open(CLINICAL_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(clinical_data, f, indent=2, ensure_ascii=False)

def transcribe_audio(audio_file):
    """
    Transcribes audio file to text using OpenAI's Whisper API
    """
    try:
        with open(audio_file, "rb") as f:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=LANGUAGE.lower() if LANGUAGE in ["LT", "RU", "EN"] else None
            )
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def extract_clinical_info(text, context):
    """
    Extracts clinical information from text with growing context
    """
    if not text or text.strip() == "":
        return {}

    try:
        language_name = SUPPORTED_LANGUAGES.get(LANGUAGE, "Lithuanian")

        # Define a schema for efficient information extraction
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "extract_clinical_information",
                    "description": f"Extract key clinical information from medical consultation in {language_name}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "utterance_type": {
                                "type": "string",
                                "enum": ["patient_statement", "doctor_question", "doctor_instruction", "irrelevant"],
                                "description": "Type of utterance - whether it's a patient describing symptoms/medical history or a doctor asking a question or giving instructions"
                            },
                            "extracted_info": {
                                "type": "object",
                                "description": "Information extracted from the patient statement",
                                "properties": {
                                    "category": {
                                        "type": "string",
                                        "enum": ["arrival_data", "referral_data", "medical_data.complaints",
                                                "medical_data.medical_history", "medical_data.health_assessment",
                                                "diagnostics", "diagnoses", "allergies", "health_risk_factors",
                                                "treatment", "other"],
                                        "description": "The category of the extracted information"
                                    },
                                    "subcategory": {
                                        "type": "string",
                                        "description": "More specific subcategory if applicable (e.g., systolic_blood_pressure)"
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "The extracted information value"
                                    },
                                    "confidence": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"],
                                        "description": "Confidence level in the extracted information"
                                    }
                                },
                                "required": ["category", "value"]
                            }
                        },
                        "required": ["utterance_type"]
                    }
                }
            }
        ]

        # Include growing context in the prompt
        context_summary = ""
        if context:
            context_summary = "Current known information:\n"
            for category, value in context.items():
                if isinstance(value, dict):
                    for subcategory, subvalue in value.items():
                        if subvalue and not isinstance(subvalue, dict):
                            context_summary += f"- {category}.{subcategory}: {subvalue}\n"
                        elif isinstance(subvalue, dict):
                            for subsubcategory, subsubvalue in subvalue.items():
                                if subsubvalue:
                                    context_summary += f"- {category}.{subcategory}.{subsubcategory}: {subsubvalue}\n"
                elif value:
                    context_summary += f"- {category}: {value}\n"

        # Make the OpenAI API call
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": f"""You are a clinical information extraction assistant for medical consultations in {language_name}.
                Your task is to analyze a single utterance and identify if it contains relevant medical information.
                
                IMPORTANT:
                1. Distinguish between utterances from the patient (describing symptoms, medical history, etc.) and 
                   utterances from the doctor (asking questions, giving instructions).
                2. Only extract information from patient statements. For doctor questions or instructions, 
                   just classify them as such without extracting information.
                3. Be very precise and only extract information explicitly mentioned in this specific utterance.
                4. Use the current known information as context to better understand new information.
                5. For information like complaints and medical history, extract ONLY NEW information not already in the context.
                6. The output will be used to update a real clinical record, so accuracy is crucial."""},
                {"role": "user", "content": f"{context_summary}\n\nAnalyze this utterance: \"{text}\""}
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "extract_clinical_information"}}
        )

        # Extract the function call results
        if response.choices[0].message.tool_calls:
            function_call = response.choices[0].message.tool_calls[0]
            extracted_data = json.loads(function_call.function.arguments)
            return extracted_data
        return {}

    except Exception as e:
        print(f"Error extracting clinical information: {e}")
        return {}

def update_clinical_data(data, extraction):
    """
    Updates the clinical data structure with new extracted information
    """
    if "utterance_type" not in extraction or extraction["utterance_type"] != "patient_statement":
        return data  # No updates for non-patient statements

    if "extracted_info" not in extraction:
        return data  # No extracted information

    info = extraction["extracted_info"]
    category = info.get("category", "")
    subcategory = info.get("subcategory", "")
    value = info.get("value", "")

    if not category or not value:
        return data  # Required fields missing

    # Handle nested categories (e.g., medical_data.complaints)
    if "." in category:
        parts = category.split(".")
        if len(parts) == 2:
            main_category, sub_category = parts
            if main_category in data and sub_category in data[main_category]:
                # Handle special text fields that should be appended, not replaced
                if sub_category in ["complaints", "medical_history", "other_assessment_info"]:
                    current = data[main_category][sub_category]
                    if current and value not in current:
                        data[main_category][sub_category] = f"{current}; {value}"
                    elif not current:
                        data[main_category][sub_category] = value
                # Handle health assessment with subcategory
                elif sub_category == "health_assessment" and subcategory:
                    if subcategory in data[main_category][sub_category]:
                        data[main_category][sub_category][subcategory] = value
                else:
                    data[main_category][sub_category] = value
    # Handle main categories
    elif category in data:
        if isinstance(data[category], dict) and subcategory:
            if subcategory in data[category]:
                data[category][subcategory] = value
        else:
            # For text fields that should be appended, not replaced
            if category in ["diagnostics", "diagnoses", "allergies", "health_risk_factors", "notes"]:
                current = data[category]
                if current and value not in current:
                    data[category] = f"{current}; {value}"
                elif not current:
                    data[category] = value
            else:
                data[category] = value

    return data

def save_audio_chunk(frames, filename):
    """
    Saves audio frames to a WAV file for processing
    """
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    p.terminate()

def record_audio():
    """
    Records audio continuously in chunks and adds them to the queue
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("* Listening for clinical information...")

    while not stop_event.is_set():
        frames = []
        for _ in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
            if stop_event.is_set():
                break
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)

        if frames:  # Only add to queue if we have recorded frames
            audio_queue.put(frames)

    print("* Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_chunks():
    """
    Processes audio chunks from the queue and extracts clinical information incrementally
    """
    global clinical_data
    global conversation_history

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            frames = audio_queue.get(timeout=1)

            # Create temporary file for the audio chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Save audio chunk to file
            save_audio_chunk(frames, temp_filename)

            # Transcribe audio to text
            transcription = transcribe_audio(temp_filename)

            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

            if transcription:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{current_time}] Transcription: {transcription}")

                # Add the transcription to conversation history with timestamp
                conversation_history.append({
                    "timestamp": current_time,
                    "text": transcription
                })

                # Extract information from this utterance with context
                extracted_info = extract_clinical_info(transcription, clinical_data)

                if extracted_info:
                    utterance_type = extracted_info.get("utterance_type", "")

                    # Only extract information from patient statements
                    if utterance_type == "patient_statement" and "extracted_info" in extracted_info:
                        print(f"[{current_time}] Patient statement detected.")

                        info = extracted_info["extracted_info"]
                        category = info.get("category", "")
                        value = info.get("value", "")

                        if category and value:
                            # Update the clinical data with new information
                            clinical_data = update_clinical_data(clinical_data, extracted_info)

                            # Save the updated data to file
                            save_clinical_data()

                            # Show what was extracted
                            print(f"\n--- Extracted Information ---")
                            print(f"Category: {category}")
                            print(f"Value: {value}")

                            # Also show the current state of relevant section
                            print("\nCurrent Clinical Data State (Relevant Section):")
                            if "." in category:
                                parts = category.split(".")
                                if len(parts) == 2:
                                    main_category, sub_category = parts
                                    if main_category in clinical_data and sub_category in clinical_data[main_category]:
                                        print(f"{main_category}.{sub_category}: {clinical_data[main_category][sub_category]}")
                            elif category in clinical_data:
                                if isinstance(clinical_data[category], dict):
                                    print(f"{category}:")
                                    for k, v in clinical_data[category].items():
                                        if v and not isinstance(v, dict):
                                            print(f"  {k}: {v}")
                                else:
                                    print(f"{category}: {clinical_data[category]}")

                    # For doctor utterances, just show the classification
                    elif utterance_type in ["doctor_question", "doctor_instruction"]:
                        print(f"[{current_time}] {utterance_type.replace('_', ' ').title()} detected.")

            # Mark the task as done in the queue
            audio_queue.task_done()

        except queue.Empty:
            # Queue is empty, wait a bit
            time.sleep(0.1)
        except Exception as e:
            print(f"Error processing audio chunk: {e}")

def handle_commands():
    """
    Handles user commands during the session
    """
    print("\nAvailable commands:")
    print("  view       - Show current clinical data")
    print("  save       - Save clinical data to file")
    print("  quit/exit  - Stop recording and exit")

    while not stop_event.is_set():
        try:
            command = input().strip().lower()

            if command in ["quit", "exit", "q"]:
                stop_event.set()
                print("Stopping recording...")

            elif command in ["view", "v"]:
                print("\nCurrent Clinical Data:")
                print(json.dumps(clinical_data, indent=2, ensure_ascii=False))

            elif command in ["save", "s"]:
                save_clinical_data()
                print("Clinical data saved to file.")

            elif command:
                print("Unknown command. Available commands: view, save, quit/exit")

        except EOFError:
            break
        except Exception as e:
            print(f"Error handling command: {e}")

def main():
    """
    Main function to run the clinical information collector
    """
    global clinical_data

    try:
        language_name = SUPPORTED_LANGUAGES.get(LANGUAGE, "Lithuanian")

        # Load existing data or initialize new data structure
        clinical_data = load_or_initialize_data()

        # Create and start recording thread
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.daemon = True
        recording_thread.start()

        # Create and start processing thread
        processing_thread = threading.Thread(target=process_audio_chunks)
        processing_thread.daemon = True
        processing_thread.start()

        # Create and start command handling thread
        command_thread = threading.Thread(target=handle_commands)
        command_thread.daemon = True
        command_thread.start()

        # Keep the main thread running
        print("=" * 70)
        print(f"Clinical information collector is running in {language_name}.")
        print(f"Default language: {LANGUAGE} - {language_name}")
        print("Information is being collected incrementally and saved in real-time.")
        print("=" * 70)
        print("Enter 'view' to show the current clinical data")
        print("Enter 'save' to manually save the data")
        print("Enter 'quit' or 'exit' to stop and exit")
        print("=" * 70)

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping clinical information collector...")
        stop_event.set()

        # Wait for threads to finish
        recording_thread.join(timeout=2)
        processing_thread.join(timeout=2)
        command_thread.join(timeout=2)

        # Save the final clinical data to a file
        save_clinical_data()

        # Also save the conversation history
        with open("conversation_history.json", "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)

        print(f"Final clinical data saved to {CLINICAL_DATA_FILE}")
        print(f"Conversation history saved to conversation_history.json")

        print("All processes stopped. Exiting.")

if __name__ == "__main__":
    main()