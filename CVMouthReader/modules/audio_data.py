import fitz  
import whisper 
import moviepy.editor as mp 
import difflib
import os

def sync_script_with_video(pdf_path, video_path, output_path="aligned_script.txt"):
    """Extracts text from a movie script PDF, transcribes the video, aligns them, and saves the output."""
    
    def extract_text_from_pdf(pdf_path):
        """Extracts text from a PDF script."""
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)

    def transcribe_video(video_path, temp_audio_path="temp_audio.wav"):
        """Transcribes audio from a video file using Whisper AI."""
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio_path, logger=None)
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path)
        os.remove(temp_audio_path)  # Clean up temporary audio file
        return result['segments']

    def align_script_with_transcription(script_text, transcription_segments):
        """Aligns extracted script lines with transcription timestamps."""
        script_lines = script_text.splitlines()
        aligned_script = []
        
        for segment in transcription_segments:
            transcript = segment['text'].strip()
            best_match = max(script_lines, key=lambda line: difflib.SequenceMatcher(None, line, transcript).ratio(), default="")
            match_ratio = difflib.SequenceMatcher(None, best_match, transcript).ratio()
            if match_ratio > 0.5:  # Threshold for match confidence
                aligned_script.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'script_line': best_match,
                    'transcript': transcript
                })
                script_lines.remove(best_match)
        
        return aligned_script

    def save_aligned_script(aligned_script, output_path):
        """Saves aligned script and timestamps to a text file."""
        with open(output_path, 'w') as f:
            for entry in aligned_script:
                f.write(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['script_line']}\n")

    # Run pipeline
    print("Extracting script text...")
    script_text = extract_text_from_pdf(pdf_path)

    print("Transcribing video...")
    transcription_segments = transcribe_video(video_path)

    print("Aligning script with transcription...")
    aligned_script = align_script_with_transcription(script_text, transcription_segments)

    print("Saving aligned script...")
    save_aligned_script(aligned_script, output_path)

    print(f"Process completed! Aligned script saved to {output_path}")

