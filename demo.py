import os
import pandas as pd
from bs4 import BeautifulSoup

def parse_review_file(file_path):

    with open(file_path, 'r') as file:
        file_content = file.read()


    soup = BeautifulSoup("<root>" + file_content + "</root>", 'html.parser')

    # Extract all DOC elements
    docs = soup.find_all('doc')
    docno = soup.find_all('docno')
    meta = parse_car_meta(docno[0].text) if docno else ("Unknown", "Unknown", "Unknown")

    # Prepare data collection
    reviews = []

    # Process each document
    for doc in docs:
        date_tag = doc.find("date")
        author_tag = doc.find("author")
        text_tag = doc.find("text")
        favorite_tag = doc.find("favorite")

        # Extract text content, handle missing tags
        date = date_tag.text if date_tag else None
        author = author_tag.text if author_tag else None
        text = text_tag.text if text_tag else None
        favorite = favorite_tag.text if favorite_tag else None

        # Add to collection
        reviews.append({
            "DATE": date,
            "AUTHOR": author,
            "TEXT": text,
            "FAVORITE": favorite,
            "MODEL": meta[2],
            "BRAND": meta[1],
            "YEAR": meta[0]
        })

    # Create DataFrame
    return pd.DataFrame(reviews)


def parse_car_meta(filename):
    parts = filename.split("_")

    year = parts[0]
    brand = parts[1]
    model = parts[2]

    return year, brand, model

# Initialize an empty list to store dataframes
dataframes = []

# Define the path template
base_path = './data/cars/'
base_path = './test/'


for year in os.listdir(base_path):
    path = os.path.join(base_path, year)
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        # Check if it's a file
        if os.path.isfile(file_path):
            # Parse the file into a DataFrame (assuming CSV format)
            df = parse_review_file(file_path)
            dataframes.append(df)


# Combine all dataframes into one
final_dataframe = pd.concat(dataframes, ignore_index=True)

# Display the combined DataFrame
print(final_dataframe)



