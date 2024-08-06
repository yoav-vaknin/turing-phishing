import pandas as pd

# Reading the fabricated users dataset
users_dataset = pd.read_csv('data/fabricated_users/SocialMediaUsersDataset.csv')
# Selecting 100 fabricated users data
randomly_selected_users = users_dataset.sample(100)
# Saving data to a CSV
randomly_selected_users.to_csv('data/fabricated_users/RandomlySampledUsers.csv')