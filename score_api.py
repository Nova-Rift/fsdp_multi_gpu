import json
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hyperparams', type=str)

args = parser.parse_args()

hyperparams_str = args.hyperparams

print("************************************************")
print(hyperparams_str)
print("************************************************")

# Read the contents of the updated JSON file
with open('output_responses.json', 'r') as file:
    message_content = json.load(file)

# Replace the placeholders in the message content for each message
for message in message_content['messages']:
    message['message'] = message['message'].replace('<|start_header_id|>', '<start_header_id>')\
                                            .replace('<|end_header_id|>', '<end_header_id>')\
                                            .replace('<|eot_id|>', '<eot_id>')\
                                            .replace('<|begin_of_text|>', '<begin_of_text>')

# Extract the list of messages (not the entire 'messages' dictionary)
message_list = message_content['messages']

# Read the data.txt file
with open('data.json', 'r') as file:
    data_content = json.load(file)

# Prepare the payload with the message list (as a list) and data content
payload = {
    "message": message_list,  # Now a list, not a dictionary with 'messages' key
    "data": json.dumps(data_content),      # Raw data content from 'data.txt'
    "hyperparams": hyperparams_str
}

# Define the URL for the POST request
url = "https://example.com/score"

# Send the POST request with the payload converted to JSON
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Print the server response
print(response.text)
