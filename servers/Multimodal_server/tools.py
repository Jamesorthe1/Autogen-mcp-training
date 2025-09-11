import folium
from geopy.geocoders import Nominatim
import requests
import polyline
import math
import tempfile
import webbrowser
from PIL import Image
from io import BytesIO
from googleapiclient.discovery import build
import os # Import os for env vars

# Hardcoded values for fallback (should ideally be in a config or env)
# These are used for Google Custom Search API if environment variables are not set.
HARDCODED_GOOGLE_CLOUD_API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY', 'AIzaSyAECC4ASkB6uOZMQXkbiNYsaHRGf1_K6k') # Use env var as first priority
HARDCODED_PROGRAMMABLE_SEARCH = os.getenv('PROGRAMMABLE_SEARCH', '848db169675dd476e') # Use env var as first priority


def display_map_for_location(location_name: str):
    """
    Generates and displays a Folium map centered at a given location.
    It uses Nominatim to geocode the location name to coordinates.
    The map is saved to a temporary HTML file and opened in the default browser.

    Args:
        location_name (str): The name of the location to display the map for.
    """
    print(f"Generating map for location: {location_name}")
    try:
        geolocator = Nominatim(user_agent="multimodal_rag_app")
        location_data = geolocator.geocode(location_name)

        if location_data:
            latitude = location_data.latitude
            longitude = location_data.longitude
            print(f"Found coordinates for '{location_name}': Latitude={latitude}, Longitude={longitude}")
        else:
            print(f"Could not find coordinates for '{location_name}'. Using default location (0,0).")
            latitude = 0
            longitude = 0

        # Create a Folium map centered at the location
        m = folium.Map(location=[latitude, longitude], zoom_start=10)
        # Add a marker for the location
        folium.Marker([latitude, longitude], popup=location_name).add_to(m)

        # Save the map to a temporary HTML file and open it in the default browser
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
            map_path = tmpfile.name
            m.save(map_path)

        print(f"Saved map to temporary file: {map_path}")
        webbrowser.open(f'file://{map_path}') # Open the HTML file
        print("Folium map displayed in a browser.")

    except Exception as e:
        print(f"An error occurred while generating or displaying the Folium map: {e}")

def display_travel_time_and_route(source_location: str, destination_location: str):
    """
    Calculates the estimated travel time and displays the route between two locations
    on a Folium map using the Open Source Routing Machine (OSRM) API.
    The map is saved to a temporary HTML file and opened in the default browser.

    Args:
        source_location (str): The starting location name.
        destination_location (str): The ending location name.
    """
    geolocator = Nominatim(user_agent="multimodal_rag_app")

    try:
        # Geocode source and destination locations to get coordinates
        source_coords = geolocator.geocode(source_location)
        destination_coords = geolocator.geocode(destination_location)

        if not source_coords or not destination_coords:
            print("Could not find coordinates for one or both locations.")
            return

        source_lon, source_lat = source_coords.longitude, source_coords.latitude
        dest_lon, dest_lat = destination_coords.longitude, destination_coords.latitude

        # Construct the OSRM API request URL for driving directions
        # Using the public OSRM demo server
        url = f"http://router.project-osrm.org/route/v1/driving/{source_lon},{source_lat};{dest_lon},{dest_lat}?overview=full&geometries=polyline"

        print(f"Requesting route from {source_location} to {destination_location} from OSRM...")
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        directions_data = response.json()

        if directions_data.get('code') == 'Ok': # Use .get() for safety
            route = directions_data['routes'][0]
            duration_seconds = route['duration'] # Duration in seconds

            # Calculate travel time in hours and minutes
            hours = math.floor(duration_seconds / 3600)
            minutes = math.floor((duration_seconds % 3600) / 60)
            travel_time = f"{hours} hours and {minutes} minutes"

            # Create a Folium map centered between the source and destination
            map_center = ((source_lat + dest_lat) / 2,
                          (source_lon + dest_lon) / 2) # Corrected map center calculation
            m = folium.Map(location=map_center, zoom_start=10)

            # Add markers for source and destination
            folium.Marker([source_lat, source_lon], popup=f"Start: {source_location}").add_to(m)
            folium.Marker([dest_lat, dest_lon], popup=f"End: {destination_location}").add_to(m)

            # Decode the polyline geometry and add it to the map
            points = polyline.decode(route['geometry'])
            folium.PolyLine(points, color="blue", weight=2.5, opacity=1).add_to(m)

            # Save the map to a temporary HTML file and open it
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                 map_path = tmpfile.name
                 m.save(map_path)

            print(f"Saved map to temporary file: {map_path}")
            webbrowser.open(f'file://{map_path}') # Open the HTML file
            print("Folium map displaying route and travel time in a browser.")

            print(f"The estimated travel time from {source_location} to {destination_location} is {travel_time}.")

        else:
            print(f"Could not find directions from {source_location} to {destination_location}. Status: {directions_data.get('code', 'Unknown')}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the OSRM request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def search_images(query: str, num: int = 5):
    """
    Searches for images using the Google Custom Search API (requires API key and CSE ID).

    Args:
        query (str): The search query string.
        num (int): The maximum number of images to return (default is 5).

    Returns:
        list: A list of image URLs found, or an empty list if the search fails or returns no results.
    """
    # Use environment variables for API keys, fallback to hardcoded if not set
    api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
    cse_id = os.getenv('PROGRAMMABLE_SEARCH')

    # Fallback to hardcoded values if environment variables are not set
    if not api_key:
         api_key = HARDCODED_GOOGLE_CLOUD_API_KEY
         print("Using hardcoded GOOGLE_CLOUD_API_KEY for image search.")
    if not cse_id:
         cse_id = HARDCODED_PROGRAMMABLE_SEARCH
         print("Using hardcoded PROGRAMMABLE_SEARCH ID for image search.")


    if not api_key or not cse_id:
        print("Google Cloud API Key or Programmable Search Engine ID not available (neither from env vars nor hardcoded). Cannot perform image search.")
        return []

    service = build("customsearch", "v1", developerKey=api_key)
    try:
        print(f"Searching for images with query: '{query}' (limit: {num})")
        res = service.cse().list(q=query, cx=cse_id, searchType='image', num=num).execute()
        if 'items' in res:
            print(f"Found {len(res['items'])} images.")
            return [item['link'] for item in res['items']]
        else:
            print("Image search returned no items.")
            return []
    except Exception as e:
        print(f"An error occurred during image search: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed traceback
        return []

def display_images(image_urls: list):
    """
    Displays images from a list of URLs using Pillow's show() method.
    Each image will be opened in the default image viewer.

    Args:
        image_urls (list): A list of image URLs (strings).
    """
    print(f"Attempting to display {len(image_urls)} images...")
    if not image_urls:
        print("No image URLs provided to display.")
        return

    for i, url in enumerate(image_urls):
        try:
            print(f"Displaying image {i+1}/{len(image_urls)} from URL: {url}")
            response = requests.get(url)
            response.raise_for_status() # Raise an HTTPError for bad responses
            img = Image.open(BytesIO(response.content))
            # Use Pillow's show() method to open the image in a default viewer
            img.show()
            print(f"Successfully displayed image from URL: {url}")
        except requests.exceptions.RequestException as req_e:
            print(f"Error fetching image from URL {url}: {req_e}")
        except Exception as e:
            print(f"Could not display image from URL {url}. Error: {e}")
