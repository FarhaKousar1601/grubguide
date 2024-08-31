from dotenv import load_dotenv
load_dotenv(".env.local")
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI   
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Optionally delete the existing index
# pc.delete_index("rag")  # Uncomment this line if you want to delete the existing index

# # Create a Pinecone index
# pc.create_index(
#     name="rag",
#     dimension=1536,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
# )

# Load the review data
data = json.load(open("uiuc_campustown_restaurants_yelp.json"))

# Function to convert hours to a readable format
def format_hours(hours):
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    formatted_hours = []
    for hour in hours:
        day = days[hour["day"]]
        start = f"{hour['start'][:2]}:{hour['start'][2:]}"
        end = f"{hour['end'][:2]}:{hour['end'][2:]}"
        formatted_hours.append(f"{day}: {start} - {end}")
    return "; ".join(formatted_hours)

def remove_non_ascii(s):
    return s.encode('ascii', 'ignore').decode('ascii')

processed_data = []
client = OpenAI()

# Create embeddings for each review
for review in data:
    review = {
        "name": remove_non_ascii(review["name"]),
        "google_rating": str(review["google_rating"]),
        "price_range": review["price_range"],
        "address": remove_non_ascii(review["address"]),
        "cuisine": remove_non_ascii(review["cuisine"]),
        "hours": remove_non_ascii(format_hours(review["hours"]))
    }
    response = client.embeddings.create(
        input=review.values(),
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    processed_data.append(
        {
            "values": embedding,
            "id": review["name"],
            "metadata": {
                "url": review["address"],
                "tags": review["cuisine"],
                "rating": review["google_rating"],
                "price_range": review["price_range"],
                "hours": review["hours"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())