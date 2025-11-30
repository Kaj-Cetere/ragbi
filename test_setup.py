import os
from dotenv import load_dotenv

load_dotenv()

print("Environment Variables:")
print(f"OPENROUTER_API_KEY: {'✓' if os.getenv('OPENROUTER_API_KEY') else '✗'}")
print(f"SUPABASE_URL: {'✓' if os.getenv('SUPABASE_URL') else '✗'}")
print(f"SUPABASE_KEY: {'✓' if os.getenv('SUPABASE_KEY') else '✗'}")

print("\nDirectories:")
dirs = [
    'Shulchan-Arukh-Cleaned-With-Metadata/cleaned_with_metadata/Orach Chayim/segments',
    'Shulchan-Arukh-Cleaned-With-Metadata/cleaned_with_metadata/Yoreh De\'ah/segments',
    'Shulchan-Arukh-Cleaned-With-Metadata/cleaned_with_metadata/Even HaEzer/segments',
    'Shulchan-Arukh-Cleaned-With-Metadata/cleaned_with_metadata/Choshen Mishpat/segments'
]

for d in dirs:
    exists = os.path.exists(d)
    if exists:
        file_count = len([f for f in os.listdir(d) if f.endswith('.json')])
        print(f"{d}: ✓ ({file_count} files)")
    else:
        print(f"{d}: ✗")
