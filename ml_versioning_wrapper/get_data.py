from azure.storage.blob import BlobServiceClient
import os


def get_data_from_storage(wrapper_azure_container_name: str,
                          wrapper_origin_file_name: str,
                          data_folder="./data/"
                          ) -> None:
    """
    download data from Azure Storage
    :param wrapper_azure_container_name: Azure Storage Container name
    :param wrapper_origin_file_name: Blob name
    :param data_folder: folder where to save data locally
    :return: None : files are save locally
    """
    if wrapper_azure_container_name:
        if wrapper_origin_file_name not in os.listdir(data_folder):
            connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            blob_service_client = BlobServiceClient.from_connection_string(
                connect_str)
            blob_client = blob_service_client.get_blob_client(
                container=wrapper_azure_container_name,
                blob=wrapper_origin_file_name)

            # Download csv file
            with open(data_folder + wrapper_origin_file_name, "wb") as \
                    download_file:
                download_file.write(blob_client.download_blob().readall())
