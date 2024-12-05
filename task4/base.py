!pip install pygithub

from github import Github
import requests
import pandas as pd

"""### Necessary Variables"""

rwanda = []
lagos = []
nig = []
rwanda_profiles = []
lagos_profiles = []
nig_profiles = []
client = Github('bba9dad9bd7ef7623ba0', '55f7e1424f078d287bbdbbe343a54fbe17592257', per_page=1000)
headers = {"Authorization": "token {}".format('')}

def get_users_by_location(list, locate):
  users = client.search_users(str('machine learning')+(f'location:{locate}'))
  print(users)
  for user in users:
    u = str(user)
    u = u.strip('NamedUser(login=)')
    u = u.strip('""')
    list.append(u)
  
def get_users_profile(list, location):

  for n in range(len(location)):
    req = requests.get(f'https://api.github.com/users/{location[n]}', headers=headers)
    list.append(req.json())

"""### Querying Users through API qualifiers"""

get_users_by_location(lagos, 'Lagos')
get_users_by_location(nig, 'Nigeria')
get_users_by_location(rwanda, 'Rwanda')

get_users_profile(lagos_profiles, lagos)
get_users_profile(nig_profiles, nig)
get_users_profile(rwanda_profiles, rwanda)

"""### Proccessing DataFrame"""

lg = pd.DataFrame(lagos_profiles)
ng = pd.DataFrame(nig_profiles)
rd = pd.DataFrame(rwanda_profiles)

df = pd.concat([lg, ng])
df = pd.concat([df, rd])
df.drop(columns=['node_id', 'gravatar_id', 'avatar_url', 'html_url', 'followers_url', 'following_url', 'gists_url',
                 'starred_url', 'subscriptions_url', 'organizations_url', 'repos_url', 'events_url', 
                 'received_events_url', 'type', 'site_admin', 'hireable', 'twitter_username', 
                 'public_repos', 'public_gists', 'created_at', 'updated_at', 'private_gists', 'total_private_repos',
                 'owned_private_repos', 'disk_usage', 'collaborators', 'two_factor_authentication', 'plan'], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Assessment/Github Data Ingestion/ml_git_users.csv')
pd.to_datetime(client.rate_limiting_resettime, unit='s')