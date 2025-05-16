import logging
import os
import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import openai, silero

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinical-data-extractor")

load_dotenv()


# Define the ClinicalRecord model
class ClinicalRecord(BaseModel):
    date: date
    time: Optional[str] = None
    physician: Optional[str] = None
    diagnosis: Optional[str] = None
    clinical_diagnosis: Optional[str] = None
    anamnesis_and_complaints: Optional[str] = None  # Nusiskundimai ir anamnezė
    objective_condition: Optional[str] = None  # Objektyvi būklė
    research_and_consultation_plan: Optional[str] = None  # Tyrimų ir konsultacijų planas
    performed_tests_and_consultations: Optional[str] = None  # Atlikti tyrimai ir konsultacijos
    treatment_applied: Optional[str] = None  # Taikytas gydymas
    medication_treatment: Optional[str] = None  # Taikytas medikamentinis gydymas
    condition_on_discharge: Optional[str] = None  # Būklė išrašant
    recommendations: Optional[str] = None  # Rekomendacijos
    notes: Optional[str] = None  # Pastabos
    disability_info: Optional[str] = None  # Nedarbingumai
    allergies: Optional[List[str]] = None  # Alergijos (list of substances)
    allergy_descriptions: Optional[List[str]] = None  # Aprašas
    allergy_dates: Optional[List[date]] = None  # Data of allergy records
    vaccinations: Optional[List[str]] = None  # Skiepai
    prescriptions: Optional[List[str]] = None  # Receptai
    referrals: Optional[List[str]] = None  # Siuntimai
    medical_certificates: Optional[List[str]] = None  # Medicininės pažymos


class ClinicalDataExtractorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a medical transcription assistant. Your role is to carefully listen to medical consultations
            and extract structured information to fill a clinical record. Pay close attention to dates, diagnoses,
            treatments, medications, and recommendations.

            As you hear information, use the update_clinical_record function to save relevant details
            into the appropriate fields. Be thorough but precise - only include information that was
            explicitly mentioned in the conversation.

            For list fields like allergies or medications, add new items as they are mentioned.
            If information is corrected during the conversation, update the record accordingly.

            Important: DO NOT prompt the speakers for specific information. Simply listen and extract
            what is naturally mentioned. If a field is not mentioned, leave it empty.
            """,
            stt=openai.STT(language='lt'),
            llm=openai.LLM(model="gpt-4.1"),
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )
        # Initialize the clinical record with today's date
        self.clinical_record = ClinicalRecord(date=date.today())
        self.conversation_transcript = ""

        # Create a transcript log file
        self.transcript_file = "consultation_transcript.txt"
        # Clear the transcript file if it exists
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"Consultation started on {date.today().isoformat()} at {datetime.now().strftime('%H:%M:%S')}\n\n")

    async def on_enter(self):
        await self.session.say(
            "I'm ready to listen to the clinical consultation. I'll extract relevant information in real-time.")
        self.session.generate_reply()

    @function_tool
    async def update_clinical_record(self, context: RunContext, field: str, value: Any):
        """
        Update a field in the clinical record with the provided value.

        Args:
            context: The current conversation context
            field: The name of the field to update in the clinical record
            value: The value to set for the field
        """
        if field in self.clinical_record.__fields__:
            # For list fields, append to the existing list or create a new one
            if field in ["allergies", "allergy_descriptions", "allergy_dates",
                         "vaccinations", "prescriptions", "referrals", "medical_certificates"]:
                current_value = getattr(self.clinical_record, field) or []
                if isinstance(value, list):
                    current_value.extend(value)
                else:
                    current_value.append(value)
                setattr(self.clinical_record, field, current_value)
            # For regular fields, just update the value
            else:
                setattr(self.clinical_record, field, value)

            logger.info(f"Updated clinical record field '{field}' with value: {value}")

            # Return confirmation message
            return None, f"Updated {field}"
        else:
            logger.warning(f"Attempted to update unknown field: {field}")
            return None, f"Error: Unknown field '{field}'"

    @function_tool
    async def get_clinical_record(self, context: RunContext):
        """
        Get the current state of the clinical record.

        Returns:
            The current clinical record as a JSON object
        """
        record_dict = self.clinical_record.dict(exclude_none=True)
        logger.info(f"Retrieved clinical record: {record_dict}")
        return None, record_dict

    @function_tool
    async def save_clinical_record(self, context: RunContext, filename: str = "clinical_record.json"):
        """
        Save the current clinical record to a JSON file.

        Args:
            context: The current conversation context
            filename: The name of the file to save the record to
        """
        # Convert the record to a JSON-serializable dict
        record_dict = self.clinical_record.dict(exclude_none=True)

        # Convert date objects to strings
        for key, value in record_dict.items():
            if isinstance(value, date):
                record_dict[key] = value.isoformat()
            elif isinstance(value, list) and all(isinstance(item, date) for item in value):
                record_dict[key] = [item.isoformat() for item in value]

        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved clinical record to {filename}")
        return None, f"Clinical record saved to {filename}"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()
    agent = ClinicalDataExtractorAgent()

    # Register event handlers for transcriptions - based on transcriber.py approach
    @session.on("user_input_transcribed")
    def on_transcript(transcript_event):
        # Log the full event to see all available metadata
        logger.debug(f"Transcript event: {transcript_event}")

        if transcript_event.is_final:
            # Append to the transcript file with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(agent.transcript_file, "a", encoding='utf-8') as f:
                f.write(f"[{timestamp}] {transcript_event.transcript}\n")

            # Update the agent's conversation transcript property
            agent.conversation_transcript += f"\n{transcript_event.transcript}"

            # Log for debugging
            logger.info(f"Final transcript: {transcript_event.transcript}")

            # Process this chunk for clinical data extraction
            # Note: The LLM will automatically process this as part of the conversation

    await session.start(
        agent=agent,
        room=ctx.room
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))