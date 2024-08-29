import pandas as pd

USER_DETAILS = '''
Name: %s
Gender: %s
Date of birth: %s
Interests: %s
City of residence: %s
Country: %s
'''

# Reading the fabricated users dataset
users_dataset = pd.read_csv(
    'data/fabricated_users/SocialMediaUsersDataset.csv')

# Selecting 100 fabricated users data, processing to string
randomly_selected_users = users_dataset.sample(100)
fewshot_selected_users = users_dataset.sample(4)

user_details = [
    USER_DETAILS % (
        user.Name,
        user.Gender,
        user.DOB,
        user.Interests,
        user.City,
        user.Country
    )
    for _, user in randomly_selected_users.iterrows()
]
user_details_df = pd.DataFrame(user_details, columns=['user_details'])
feshot_user_details_df = pd.DataFrame(user_details, columns=['user_details'])

# Saving dataframes to CSV
user_details_df.to_csv(
    'data/fabricated_users/RandomlySampledUsers.csv')
feshot_user_details_df.to_csv('data/fabricated_users/RandomlySampledUsers.csv')
