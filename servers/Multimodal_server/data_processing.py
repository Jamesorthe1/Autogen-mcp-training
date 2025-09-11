import os
from PIL import Image
import cv2
from pydub import AudioSegment
from pdfminer.high_level import extract_text

def load_multimodal_data_from_directory(data_directory):
    """
    Loads all files from the specified directory, determines their type by extension,
    and prepares them for further processing. No subfolders required.

    Args:
        data_directory (str): The path to the directory containing multimodal data files.

    Returns:
        list: A list of dictionaries, where each dictionary represents a loaded data item
              with 'id', 'type', and modality-specific content (e.g., 'text', 'image', 'video_path', 'audio_path').
    """
    print(f"Loading data from directory: {data_directory}")

    multimodal_data = []

    # Supported file extensions for each modality
    text_exts = {'.txt'}
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_exts = {'.mp3', '.wav', '.aac', '.flac'}
    pdf_exts = {'.pdf'}

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory not found at {data_directory}. Please check the path and ensure it exists.")
        return multimodal_data

    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in text_exts:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                multimodal_data.append({'id': f'text_{filename}', 'type': 'text', 'text': text_content})
                print(f"Loaded text file: {filename}") # Added print for loaded file

            elif ext in pdf_exts:
                try:
                    extracted_text = extract_text(filepath)
                    multimodal_data.append({'id': f'pdf_{filename}', 'type': 'text', 'text': extracted_text})
                    print(f"Extracted text from PDF '{filename}'")
                except Exception as e:
                    print(f"Error extracting text from PDF {filename}: {e}")

            elif ext in image_exts:
                try:
                    img = Image.open(filepath)
                    multimodal_data.append({'id': f'image_{filename}', 'type': 'image', 'image': img})
                    print(f"Loaded image file: {filename}") # Added print for loaded file
                except Exception as e:
                    print(f"Error loading image file {filename}: {e}")

            elif ext in video_exts:
                multimodal_data.append({'id': f'video_{filename}', 'type': 'video', 'video_path': filepath})
                print(f"Loaded video file: {filename}") # Added print for loaded file

            elif ext in audio_exts:
                multimodal_data.append({'id': f'audio_{filename}', 'type': 'audio', 'audio_path': filepath})
                print(f"Loaded audio file: {filename}") # Added print for loaded file

            else:
                print(f"Unsupported file type for '{filename}', skipping.")

        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

    print("\n--- Loaded Data Summary ---")
    print(f"Loaded {len([d for d in multimodal_data if d.get('type')=='text'])} text documents (including PDFs).") # Corrected summary
    print(f"Loaded {len([d for d in multimodal_data if d.get('type')=='image'])} images.")
    print(f"Loaded {len([d for d in multimodal_data if d.get('type')=='video'])} videos.")
    print(f"Loaded {len([d for d in multimodal_data if d.get('type')=='audio'])} audio files.")
    print(f"\nTotal items in multimodal_data list: {len(multimodal_data)}")


    return multimodal_data

def process_multimodal_data(multimodal_data, FRAMES_PER_SECOND=1, AUDIO_SEGMENT_LENGTH_MS=5000):
    """
    Processes raw multimodal data by extracting video frames and audio segments.
    Text and image data are passed through directly.

    Args:
        multimodal_data (list): A list of dictionaries representing loaded data items.
        FRAMES_PER_SECOND (int): The target number of frames to extract per second from videos.
        AUDIO_SEGMENT_LENGTH_MS (int): The length of audio segments in milliseconds.

    Returns:
        list: A list of dictionaries representing processed data items (text, image, video frames, audio segments).
    """
    print("Processing multimodal data...")
    processed_multimodal_data = []

    for item in multimodal_data:
        item_type = item.get('type') # Use .get() for safer access
        item_id = item.get('id') # Use .get() for safer access

        if item_type == 'text':
            if 'text' in item and item['text']:
                 processed_multimodal_data.append(item)
                 print(f"Processed text item: {item_id}") # Added print for processed item
            else:
                print(f"Skipping processing for empty text item {item_id}.")

        elif item_type == 'image':
            if 'image' in item and isinstance(item['image'], Image.Image):
                 processed_multimodal_data.append(item)
                 print(f"Processed image item: {item_id}") # Added print for processed item
            else:
                 print(f"Skipping processing for invalid image item {item_id}.")

        elif item_type == 'video':
            video_path = item.get('video_path') # Use .get()
            if not video_path:
                print(f"Skipping video item {item_id}: video_path is missing.")
                continue

            print(f"Processing video file: {video_path}")
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                     print(f"Error: Could not open video file {video_path}")
                     continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                # Ensure frame_interval is at least 1 to avoid division by zero or processing too many frames
                frame_interval = max(int(fps / FRAMES_PER_SECOND), 1) if fps > 0 else 1
                count = 0
                frames_extracted = 0 # Counter for extracted frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        processed_multimodal_data.append({'id': f'{item_id}_frame_{count:06d}', 'type': 'video_frame', 'video_id': item_id, 'frame_index': count, 'image': img})
                        frames_extracted += 1 # Increment extracted frame count
                    count += 1
                cap.release()
                cv2.destroyAllWindows()
                print(f"Extracted {frames_extracted} frames from video {item_id}.") # Report extracted frames
            except Exception as e:
                print(f"Error processing video file {video_path}: {e}")

        elif item_type == 'audio':
            audio_path = item.get('audio_path') # Use .get()
            if not audio_path or not os.path.exists(audio_path):
                print(f"Skipping audio item {item_id}: audio_path is missing or invalid.")
                continue

            print(f"Processing audio file: {audio_path}")
            try:
                audio = AudioSegment.from_file(audio_path)
                duration_ms = len(audio)
                segments_processed = 0 # Counter for processed segments
                for i in range(0, duration_ms, AUDIO_SEGMENT_LENGTH_MS):
                    segment = audio[i:i + AUDIO_SEGMENT_LENGTH_MS]
                    # Note: 'audio_segment_data' contains the pydub AudioSegment object.
                    # The embedding model will need the file path or raw data for OpenL3.
                    # Passing the path again here for potential use in embedding generation.
                    processed_multimodal_data.append({'id': f'{item_id}_segment_{i:09d}', 'type': 'audio_segment', 'audio_id': item_id, 'start_time_ms': i, 'end_time_ms': i + len(segment), 'audio_path': audio_path, 'audio_segment_data': segment})
                    segments_processed += 1 # Increment processed segment count
                print(f"Processed {segments_processed} segments from audio {item_id}.") # Report processed segments
            except Exception as e:
                print(f"Error processing audio file {audio_path}: {e}")

    print(f"\nFinished processing multimodal data. Total items for embedding: {len(processed_multimodal_data)}")
    return processed_multimodal_data