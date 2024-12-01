import pandas as pd

# Load the CSV file
df = pd.read_csv('Phishing_email.csv')

# Remove rows where either 'body' or 'label' is blank (NaN or empty string)
df_cleaned = df.dropna(subset=['body', 'label'])  # Remove rows with NaN values
df_cleaned = df_cleaned[(df_cleaned['body'] != '') & (df_cleaned['label'] != '')]  # Remove rows with empty strings

# Save the cleaned DataFrame back to CSV
df_cleaned.to_csv('Phishing_email.csv', index=False)

print("Rows with blank 'body' or 'label' have been removed.")