import requests

# URL for "The Complete Works of William Shakespeare" (Plain Text UTF-8)
url = "https://www.gutenberg.org/files/100/100-0.txt"

# Download the content
response = requests.get(url)
if response.status_code == 200:
    with open("shakespeare_complete.txt", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("Downloaded and saved 'The Complete Works of William Shakespeare'.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")