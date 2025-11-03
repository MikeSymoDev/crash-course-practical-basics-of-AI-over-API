"""This script uses the Google Gemini API to process images and extract information from them.
The script saves the extracted information to a TXT with
the same name as the image file. The script processes multiple images in a batch."""

# Import the required libraries
import json
import os
import re
import time
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Import the Google Gemini client
import google.generativeai as genai

# Save the start time, set the image and output directories
start_time = time.time()
total_files = 0
total_in_tokens = 0
total_out_tokens = 0
input_cost_per_mio_in_dollars = 2.5
output_cost_per_mio_in_dollars = 10

image_directory = "../image_data"
output_directory = "../answers/google"

# Clear the output directory
for root, _, filenames in os.walk(output_directory):
    for filename in filenames:
        os.remove(os.path.join(root, filename))

# Set the API key, model, section, and temperature
api_key = os.getenv("GEMINI_API_KEY")
model_name = "gemini-2.5-flash"
temperature = 0.5

if not api_key:
    raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte .env Datei prüfen!")

# Configure the GenerativeAI client with your API key.
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)

# Process each image in the image_data directory
for root, _, filenames in os.walk(image_directory):
    file_number = 1
    total_files = len(filenames)
    for filename in filenames:
        if filename.endswith(".jpg"):
            print("----------------------------------------")
            print(f"> Processing file ({file_number}/{total_files}): {filename}")
            image_id = filename.split(".")[0]

            # Create the prompt for the model
            print("> Sending the image to the API and requesting answer...", end=" ")
            prompt = ('Transkribiere mir den Text auf diesem Bild. '
            'Der Text ist in Fraktur geschrieben. '
            'Es handelt sich um einen Text von 1918, es geht um die Vorarlberger Frage. '
            'Gebe mir den Text Wort für Wort wieder.')

            image_path = os.path.join(root, filename)
            image = Image.open(image_path)

            answer = model.generate_content([prompt, image])
            print("Done.")

            # Extract the answer from the response
            answer_text = answer.text
            print("> Received an answer from the API. Token cost (in/out):",
                  answer.usage_metadata.prompt_token_count, "/",
                  answer.usage_metadata.candidates_token_count)
            total_in_tokens += answer.usage_metadata.prompt_token_count
            total_out_tokens += answer.usage_metadata.candidates_token_count

            print("> Processing the answer...")
            
            # Save answer as txt
            os.makedirs(output_directory, exist_ok=True)
            
            with open(f"{output_directory}/{image_id}.txt", "w", encoding="utf-8") as txt_file:
                txt_file.write(answer_text)
                print(f"> Saved the answer for {image_id} to {output_directory}/{image_id}.txt")

            # File complete: Increment the file number
            file_number += 1
            print("> Processing the answer... Done.")

# Calculate and print the total processing time
end_time = time.time()
total_time = end_time - start_time
print("----------------------------------------")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Total token cost (in/out): {total_in_tokens} / {total_out_tokens}")
print(f"Average token cost per image: {total_out_tokens / total_files}")
print(f"Total cost (in/out): ${total_in_tokens / 1e6 * input_cost_per_mio_in_dollars:.2f} / "
      f"${total_out_tokens / 1e6 * output_cost_per_mio_in_dollars:.2f}")
print("----------------------------------------")