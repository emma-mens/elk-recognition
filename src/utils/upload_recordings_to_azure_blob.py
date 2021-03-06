import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')


STREAM_DIR = "/home/pi/Videos/elk-project/"

def reformat_filename(f):
    "reformat file name to make blob subdirectory organized by date"
    fmt = '.mp4' if 'mp4' in f else '.jpg' 
    device, date, time = f.split(fmt)[0].split('_')
    date = date.replace("-", "/")
    time = "-".join(time.split("-")[:2]) # only use hour-minute resolution
    return device, date + "/" + device + "_" + time + fmt

# Get all outstanding mp4 files
files_to_upload = list(filter(lambda x: "mp4" in x or "jpg" in x, os.listdir(STREAM_DIR)))

try:
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
   
except Exception as ex:
    print('Exception:')
    print(ex)
    # Exit..  try again later
    exit(1)

for local_file_name in files_to_upload:
    try:
        upload_file_path = os.path.join(STREAM_DIR, local_file_name)
        device_name, blob_name = reformat_filename(local_file_name)
        blob_client = blob_service_client.get_blob_client(container=device_name, blob=blob_name)    

        print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)
        with open('/home/pi/gitdir/elk-recognition/src/utils/test.txt', 'a') as lf:
            lf.write(local_file_name)
        
        # Upload the video file
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data)

        # upload successful. delete local file
        os.remove(upload_file_path)
    except Exception as ex:
        print('Exception:')
        print(ex)
        # Skip.. try again next round to upload file
