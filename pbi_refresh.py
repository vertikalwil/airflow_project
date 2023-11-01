import requests
import msal

app_id = 'ce856fb3-3536-4bd0-8d56-xxxxxx'
tenant_id = 'f261b9e6-0350-42d3-ab2d-xxxxx'
username = 'xxxxx@domain'
password = '********'

authority_url = 'https://login.microsoftonline.com/' + tenant_id
scopes = ['https://analysis.windows.net/powerbi/api/.default']

client = msal.PublicClientApplication(app_id, authority=authority_url)
response = client.acquire_token_by_username_password(username = username, password = password, scopes=scopes)
access_id = response.get('access_token')

data_id = '28a3282d-8d05-489b-xxxxx-xxxx'
endpoint = f'https://api.powerbi.com/v1.0/myorg/datasets/{data_id}/refreshes'
headers = {
    'Authorization' : f'Bearer ' + access_id
}

req = requests.post(endpoint, headers=headers)
print(f'RESPONSE STATUS : {req}')