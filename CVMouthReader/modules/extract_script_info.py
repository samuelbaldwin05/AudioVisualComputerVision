import os
import csv
import re
from dotenv import load_dotenv
import PyPDF2
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to clean text (remove unnecessary punctuation outside of words)
def clean_text(text):
    """Clean the text by removing punctuation outside of words."""
    text = re.sub(r'[^a-zA-Z0-9\s\'’-]', '', text)  # Remove non-alphanumeric characters except spaces and apostrophes
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Function to split text into chunks
def split_text_into_chunks(text, num_chunks=3):
    """Split text into smaller chunks to fit within the token limits."""
    # Calculate the approximate length of each chunk
    chunk_size = len(text) // num_chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks[:num_chunks]  # Ensure only the specified number of chunks is returned

# Function to interact with OpenAI and extract the required data
def run_gpt(input):
    """Use OpenAI to extract character names, words, scene numbers, and camera cuts."""
    completion = client.chat.completions.create(
        model="gpt-4",  # Ensure you are using GPT-4 or a suitable model
        store=True,
        messages=[{"role": "user", "content": input}]
    )

    content = completion.choices[0].message.content
    return content.strip()

# Function to extract script data from OpenAI
def extract_script_data_from_openai(text_chunk, prompt):
    """Use the run_gpt function to interact with OpenAI and get the extracted data."""
    # Prepare the input for OpenAI's GPT model
    input_text = prompt + "\n\n" + text_chunk
    extracted_data = run_gpt(input_text)
    return extracted_data

# Function to write the extracted data to a CSV file
def write_to_csv(data, output_csv_path):
    """Write the extracted data to a CSV file."""
    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Main function to process the PDF, call OpenAI, and save results
def process_pdf_and_extract_data(pdf_path, output_csv_path, prompt, num_chunks=3):
    """Process the PDF and generate a CSV with extracted data."""
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Clean the extracted text
    cleaned_text = clean_text(pdf_text)

    # Step 3: Split the cleaned text into smaller chunks
    chunks = split_text_into_chunks(cleaned_text, num_chunks)

    # Step 4: Extract data from OpenAI for each chunk and save it to the CSV
    for chunk in chunks:
        extracted_data = extract_script_data_from_openai(chunk, prompt)
        
        # Assuming the extracted data is returned as rows (comma-separated)
        # Split the result into rows and write them to the CSV
        rows = [row.split(', ') for row in extracted_data.split('\n')]
        write_to_csv(rows, output_csv_path)

# Example usage
pdf_file_path = 'CVMouthReader/data/input/500-days-of-summer-2009.pdf'  # Specify the path to the PDF script file
csv_output_path = 'CVMouthReader/data/output/output_script.csv'  # Specify where to save the output CSV
openai_prompt = """
Extract the following data from the script:

- Word: The spoken word.
- Character: The name of the character saying the word.
- Scene Number: The scene number, starting from 1.
- Camera Cut Number: If there is a camera cut within a scene, track it.

For example, if the script is:
TOM
What are you doing here?

The output should be:
Word, Character, Scene Number, Camera Cut Number
What, TOM, 1, 0
are, TOM, 1, 0
you, TOM, 1, 0
doing, TOM, 1, 0
here, TOM, 1, 0
"""

# Run the process
process_pdf_and_extract_data(pdf_file_path, csv_output_path, openai_prompt, num_chunks=20)
