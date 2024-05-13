from pyyoutube import Api
from pytube import YouTube
from dotenv import dotenv_values

secrets = dotenv_values(".env")

# reference
# https://www.geeksforgeeks.org/download-youtube-videos-or-whole-playlist-with-python/

api = Api(api_key=secrets['API_KEY'])
playlist_items_by_id = api.get_playlist_items(playlist_id="PLCF7983F60455753E",
                                              count=None,
                                              return_json=True)

for item in playlist_items_by_id['items']:
    if item['status']['privacyStatus'] == 'public':
        item_id = item['contentDetails']['videoId']
        link = f'https://www.youtube.com/watch?v={item_id}'
        yt_obj = YouTube(link)
        filters = yt_obj.streams.filter(progressive=True, file_extension='mp4')
        filters.get_highest_resolution().download()
