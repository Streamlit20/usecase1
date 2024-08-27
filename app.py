from pydub import AudioSegment
import tempfile
import numpy as np
import requests
import re
import os
import time
from multiprocessing import Pool
import streamlit as st
import google.generativeai as genai

# Configure Google API key
GOOGLE_API_KEY = 'AIzaSyBwuhgihPcTMcMA8s3i9suv7TePcwESLlA'
genai.configure(api_key=GOOGLE_API_KEY)

class Document:
    def __init__(self, content):
        self.content = content
        self.metadata = {}

    @property
    def page_content(self):
        return self.content

def clean_text(text):
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'#', '', text)
    return text

def search_web(query):
    api_key = 'AIzaSyCc2_65xCxEKxEn_vWpaPvgQACmCSeX3MY'
    cse_id = '73cfc7126c4ef425c'
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}'
    response = requests.get(url)
    search_results = response.json()
    snippets = [item['snippet'] for item in search_results.get('items', [])]
    return ' '.join(snippets)

def extract_audio(media_file):
    file_extension = media_file.name.split('.')[-1].lower()
    video_extensions = ['mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'mpeg', 'mpg']
    audio_extensions = ['mp3', 'wav', 'ogg', 'm4a', 'aac', 'mpeg', 'mpg']

    if file_extension in video_extensions + audio_extensions:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            temp_file.write(media_file.read())
            media_path = temp_file.name

        try:
            audio = AudioSegment.from_file(media_path)
            audio_path = tempfile.mktemp(suffix='.mp3')
            audio.export(audio_path, format='mp3')
        finally:
            os.remove(media_path)
            return audio_path

    return None

def process_chunk_with_retry(args, max_retries=5, retry_delay=30):
    audio_chunk_path, prompt = args
    for attempt in range(max_retries):
        try:
            audio_file = genai.upload_file(path=audio_chunk_path)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content([prompt, audio_file])

            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'safety_ratings'):
                return f"Response blocked due to safety ratings: {response.safety_ratings}"
            else:
                return "No valid content returned, check model parameters or input."

        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "504" in error_message:
                print(f"API error on attempt {attempt + 1}: {error_message}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential back-off
            else:
                return f"Unhandled error: {error_message}"

    return "Failed after multiple attempts due to rate limit or other errors."

def generate_summary_and_transcript(audio_file_path):
    audio_file = genai.upload_file(path=audio_file_path)
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    summary_response = model.generate_content(["Please generate the summary of the following audio. The Summary should include in brief the topic of the call, key points discussed, any factual data mentioned, action points for respective stakeholders, and meeting conclusion of the Audio.", audio_file])
    summary = summary_response.text if hasattr(summary_response, 'text') else "Content attribute missing."

    chunk_paths = split_audio(audio_file_path)
    transcript_args = [(chunk, "Please transcribe the following audio conversation also include the names of the speakers and don't mention any time duration of the speakers.") for chunk in chunk_paths]
    with Pool() as pool:
        transcripts = pool.map(process_chunk_with_retry, transcript_args)

    for chunk_path in chunk_paths:
        os.remove(chunk_path)

    combined_transcript = "\n\n".join(transcripts)
    return summary, combined_transcript

def split_audio(audio_file_path, chunk_duration=900):
    audio = AudioSegment.from_file(audio_file_path)
    audio_duration = len(audio) / 1000  # Duration in seconds
    chunks = []

    for start in np.arange(0, audio_duration, chunk_duration):
        end = min(start + chunk_duration, audio_duration)
        chunk = audio[start * 1000:end * 1000]
        chunk_path = tempfile.mktemp(suffix='.mp3')
        chunk.export(chunk_path, format='mp3')
        chunks.append(chunk_path)
    
    return chunks

def get_gemini_response(context, question):
    """Generates content using the Gemini model."""
    try:
        # Combine context and question for the prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate text using the Gemini model
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content([prompt])
        
        # Extract the generated text from the response
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Sorry, I couldn't find the answer."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_faqs_from_transcript(transcript):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"Based on the following transcript, generate a list of frequently asked questions along with their answers. Each FAQ should be in the format:\n\nQuestion: [Your question]\nAnswer: [The answer]\n\nTranscript:\n{transcript}\n\nFAQs:"
    response = model.generate_content([prompt])
    faqs = response.text if hasattr(response, 'text') else "Content attribute missing."
    return faqs

def get_css():
    return """
    <style>
        .banner {
            position: relative;
            width: 90%;
            height: 100px;
            background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZJkPpcjFpNhfvdUcHdSytP1-tePz8v8X34Q&s');
            background-size: cover;
            color: transparent;
            text-align: center;
            line-height: 100px;
            transition: color 0.5s ease;
        }
        .banner:hover {
            color: white;
        }
        .logo {
            position: fixed;
            top: 0;
            right: 0;
            margin-right: 10px;
            margin-top: 10px;
            width: 140px;
            height: 100px;
        }
        body {
            background-image: url('https://cdn.crn.in/wp-content/uploads/2021/03/08182454/GettyImages-1140691167-scaled.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #f4f4f9;
            color: #333;
        }
        .stTextArea textarea, .stTextInput textarea {
            text-align: justify;
            background-color: #ffffff;
            border: none;
            box-shadow: none;
            resize: none;
        }
        .stTextArea, .stTextInput {
            background-color: #ffffff;
            color: #333;
            border: none;
        }
        .stButton>button {
            background-color: #31333F;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #ffffff;
        }
        .justified-text {
            text-align: justify;
        }
        .bold-title {
            font-weight: bold;
            margin-bottom: 0.1rem;
        }
        .sidebar-image {
            width: 100px;
            height: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """

st.set_page_config(page_title='Video and Audio Insight Generator', page_icon='🎥')

def upload_and_get_insights(media_file):
    audio_file_path = extract_audio(media_file)
    if audio_file_path:
        start_time = time.time()
        summary, transcript = generate_summary_and_transcript(audio_file_path)
        elapsed_time = time.time() - start_time
        num_requests = len(transcript.split()) // 300 + 1  # Estimate based on 300 words per request
        total_words = len(summary.split()) + len(transcript.split())
        os.remove(audio_file_path)
        st.session_state['insights'] = {"summary": clean_text(summary), "transcript": clean_text(transcript), "elapsed_time": elapsed_time, "num_requests": num_requests, "total_words": total_words}
    else:
        st.session_state['insights'] = {"error": "Unsupported file type or processing error."}

def ask_chatbot(context, question):
    response = get_gemini_response(context, question)
    return {"response": clean_text(response)}

def generate_faqs(transcript):
    faqs = generate_faqs_from_transcript(transcript)
    return {"faqs": clean_text(faqs)}

def app():
    st.markdown(get_css(), unsafe_allow_html=True)
    st.markdown("<div class='banner'></div>", unsafe_allow_html=True)
    st.markdown("""
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAABC1BMVEX////tGyQAAADpAAD///7vGSTi4uL74eDsAAT5ub3KysrsAADsHCX8/Pz7///u7u6Li4vc3NzDw8NPT0/87O3rFR3wOUOvr69JSUmlpaXW1tYdHR1kZGRksLCz53uTsIizpQULxV13sLTXwXGfoLiz1dnT+9/z7ys30kJPuYF34vrrxa3XyaW/xgof0gIj0tbH0pKv0kY/uTk79trvxlZnzQEz9ytH2qaXyfXf+2eDyiIT97PFbAACuZmfalZpNAADuhc4IAAAQTklEQVR4nO1cC1vbONaWLZsYZCkhxKSEW4iBYK6huC0NO00ohUJpO7Pb7n7f//8l3zlH8i3025ltzQw7o/eZp4ll3V6do3ORwjBmYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYVHC/wFWhwm/A9FODQAAAABJRU5ErkJggg==" class="logo">
        """, unsafe_allow_html=True)
    st.title("The Insight Generation")

    media_file = st.file_uploader("Upload a Video or Audio File", type=['mp4', 'mp3', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'wav', 'ogg', 'm4a', 'aac'])
    
    generate_button = st.button("Generate Insights")
    
    if generate_button and media_file:
        with st.spinner("Processing..."):
            upload_and_get_insights(media_file)

        insights = st.session_state['insights']

        if "error" not in insights:
            summary = clean_text(insights["summary"])
            transcript = clean_text(insights["transcript"])

            st.session_state['summary'] = summary
            st.session_state['transcript'] = transcript

            st.subheader("Summary:")
            st.markdown(get_css(), unsafe_allow_html=True)
            st.markdown('<p class="bold-title">Edit Summary:</p>', unsafe_allow_html=True)
            edited_summary = st.text_area("", summary, height=300)

            st.subheader("Frequently Asked Questions:")
            if 'faqs' not in st.session_state:
                with st.spinner("Generating FAQs..."):
                    faqs_response = generate_faqs(transcript)
                    if "error" not in faqs_response:
                        faqs = clean_text(faqs_response["faqs"])
                        st.session_state['faqs'] = faqs
                        if faqs.strip():
                            st.markdown(get_css(), unsafe_allow_html=True)
                            st.markdown('<p class="bold-title">Edit FAQs:</p>', unsafe_allow_html=True)
                            edited_faqs = st.text_area("", faqs, height=300)
                        else:
                            st.write("No FAQs generated from the transcript.")
                            edited_faqs = ""
                    else:
                        st.error(faqs_response["error"])
                        edited_faqs = ""
            else:
                st.markdown(get_css(), unsafe_allow_html=True)
                st.markdown('<p class="bold-title">Edit FAQs:</p>', unsafe_allow_html=True)
                edited_faqs = st.text_area("", st.session_state['faqs'], height=300)

            if st.button("Save Changes"):
                st.session_state['summary'] = edited_summary
                st.session_state['faqs'] = edited_faqs
                st.success("Changes saved successfully.")

            st.subheader("Transcript:")
            st.markdown(f"<div class='justified-text'>{transcript}</div>", unsafe_allow_html=True)

            combined_content = f"Summary:\n{edited_summary}\n\nFrequently Asked Questions:\n{edited_faqs}\n\nTranscript:\n{transcript}"
            original_filename = os.path.splitext(media_file.name)[0]
            downloadable_filename = f"{original_filename}_summary_faqs_transcript.txt"

            st.download_button("Download Combined Content", combined_content, downloadable_filename, "text/plain")


            with st.sidebar:
                st.header("Chatbot")
                question = st.text_input("Ask a question:")

                if st.button("Send", key="sendButton"):
                    with st.spinner("Processing..."):
                        chatbot_response = ask_chatbot(f"{edited_summary} {transcript}", question)

                    if "error" not in chatbot_response:
                        st.markdown(f"<div class='justified-text'>{chatbot_response['response']}</div>", unsafe_allow_html=True)
                    else:
                        st.error(chatbot_response["error"])
        else:
            st.error(f"Error: {insights['error']}")

    elif 'summary' in st.session_state and 'transcript' in st.session_state:
        summary = st.session_state['summary']
        transcript = st.session_state['transcript']

        st.subheader("Summary:")
        st.markdown(get_css(), unsafe_allow_html=True)
        st.markdown('<p class="bold-title">Edit Summary:</p>', unsafe_allow_html=True)
        edited_summary = st.text_area("", summary, height=300)

        st.subheader("Frequently Asked Questions:")
        if 'faqs' not in st.session_state:
            with st.spinner("Generating FAQs..."):
                faqs_response = generate_faqs(transcript)
                if "error" not in faqs_response:
                    faqs = clean_text(faqs_response["faqs"])
                    st.session_state['faqs'] = faqs
                    if faqs.strip():
                        st.markdown(get_css(), unsafe_allow_html=True)
                        st.markdown('<p class="bold-title">Edit FAQs:</p>', unsafe_allow_html=True)
                        edited_faqs = st.text_area("", faqs, height=300)
                    else:
                        st.write("No FAQs generated from the transcript.")
                        edited_faqs = ""
                else:
                    st.error(faqs_response["error"])
                    edited_faqs = ""
        else:
            st.markdown(get_css(), unsafe_allow_html=True)
            st.markdown('<p class="bold-title">Edit FAQs:</p>', unsafe_allow_html=True)
            edited_faqs = st.text_area("", st.session_state['faqs'], height=300)

        if st.button("Save Changes"):
            st.session_state['summary'] = edited_summary
            st.session_state['faqs'] = edited_faqs
            st.success("Changes saved successfully.")

        st.subheader("Transcript:")
        st.markdown(f"<div class='justified-text'>{transcript}</div>", unsafe_allow_html=True)

        combined_content = f"Summary:\n{edited_summary}\n\nFrequently Asked Questions:\n{edited_faqs}\n\nTranscript:\n{transcript}"
        original_filename = "combined_content"
        downloadable_filename = f"{original_filename}_summary_faqs_transcript.txt"

        st.download_button("Download Combined Content", combined_content, downloadable_filename, "text/plain")


        with st.sidebar:
            st.header("Chatbot")
            question = st.text_input("Ask a question:")

            if st.button("Send", key="sendButton"):
                with st.spinner("Processing..."):
                    chatbot_response = ask_chatbot(f"{edited_summary} {transcript}", question)

                if "error" not in chatbot_response:
                    st.markdown(f"<div class='justified-text'>{chatbot_response['response']}</div>", unsafe_allow_html=True)
                else:
                    st.error(chatbot_response["error"])

    with st.sidebar:
        st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRViRfutvNG9i9GtCPAC6qiwcK_uIOvKU0QP-zvFl3iMHKpUvAvStpetXH8o2AQ_fA4tBg&usqp=CAU", use_column_width=False, width=100)
        st.subheader("Video and Audio Insight Generator")

if __name__ == "__main__":
    app()
