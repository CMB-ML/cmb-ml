"""
This module provides a BoxInterface class for interacting with the Box API and performing file upload operations.

The BoxInterface class allows authentication with Box using JWT authentication, walk through the root folder and its child items, upload files to a specified folder, check if a file already exists in a folder, ensure that a file has a shared link, and check the current rate limit of the Box API.

Additionally, the module provides a safe_api_call function that can be used to make API calls with retries on rate limit errors.

Example usage:
    box = BoxInterface(auth_path='path/to/auth.json', root_folder_id='0')
    box.walk_root_and_child()
    box.upload_file(file_path='path/to/file.txt', folder_id='123456789')
    box.check_rate_limit()

    result = safe_api_call(lambda: box.upload_file(file_path='path/to/file.txt', folder_id='123456789'))
"""

from typing import Union, Any, Callable
import time
import logging
from pathlib import Path

from boxsdk import Client, JWTAuth
from boxsdk import object as box_object
from boxsdk.exception import BoxAPIException


logger = logging.getLogger(__name__)


class BoxInterface:
    def __init__(self, auth_path: str, root_folder_id: str='0', verbose: bool=True) -> None:
        """
        Initialize a BoxInterface object.

        Args:
            auth_path (str): The path to the JSON file containing the JWT authentication settings.
            root_folder_id (str, optional): The ID of the root folder. Defaults to '0'.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to True.
        """
        self.root_folder_id = root_folder_id
        self.verbose = verbose
        auth = JWTAuth.from_settings_file(auth_path)
        self.client = Client(auth)

    def walk_root_and_child(self) -> None:
        """
        Walk through the root folder and its child items, and print their details.
        """
        try:
            root_folder = self.client.folder(folder_id=self.root_folder_id).get()
            logger.info(f"Root folder: {root_folder.name}")
            items = root_folder.get_items(limit=1000)
            for item in items:
                logger.info(f"{item.type.capitalize()} - {item.id} - {item.name}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def walk_folders_with_action(self, folder_id: str, action: Callable) -> None:
        """
        Walk through the folders in the specified folder and perform the specified action 
        on each item within.

        Args:
            folder_id (str): The ID of the folder to walk through.
            action (function): The action to perform on each folder.
        """
        folder_structure = {}
        try:
            folder = self.client.folder(folder_id=folder_id).get()
            items = folder.get_items(limit=1000)
            items = list(items)
            for item in items:
                if item.type == 'folder':
                    folder_structure[item.name] = self.walk_folders_with_action(item.id, action)
                elif item.type == 'file':
                    folder_structure[item.name] = action(item)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        return folder_structure

    def make_shared_link_dict(self, item: box_object.file.File) -> dict:
        """
        Create a dictionary containing the details of a file's shared link.

        Args:
            item (boxsdk.object.file.File): The file object.

        Returns:
            dict: A dictionary containing the details of the shared link (file_name, box_id, shared_link).
        """
        shared_link = self.get_shared_link(item)
        return {'file_name': item.name, 'box_id': item.id, 'shared_link': shared_link}


    def make_forced_shared_link_dict(self, item: box_object.file.File) -> dict:
        """
        Force creation of new shared links for an asset.

        Args:
            item (boxsdk.object.file.File): The file object.

        Returns:
            dict: A dictionary containing the details of the shared link (file_name, box_id, shared_link).
        """
        shared_link = self.force_new_shared_link(item)
        return {'file_name': item.name, 'box_id': item.id, 'shared_link': shared_link}


    def upload_file(self, file_path: Path, 
                    folder_id: str=None, 
                    force_links: bool=True,
                    force_replace: bool=True) -> None:
        """
        Upload a file to the specified folder.

        Args:
            file_path (str): The path to the file to be uploaded.
            folder_id (str): The ID of the folder where the file should be uploaded.

        Returns:
            dict or None: A dictionary containing the details of the uploaded file (file_name, box_id, shared_link),
                          or None if the upload failed.
        """
        file_name = file_path.name

        if folder_id is None:
            folder_id = self.root_folder_id

        existing_file = self.file_exists(folder_id, file_name, with_details=True)
        if existing_file:
            if force_replace:
                logging.info(f"File {file_name} already exists in Box with ID {existing_file.id}. Replacing...")
                existing_file.delete()
            else:
                logging.info(f"File {file_name} already exists in Box with ID {existing_file.id}")
                # Check if the file has a shared link and retrieve or create if necessary
                if force_links:
                    shared_link = self.force_new_shared_link(existing_file)
                else:
                    shared_link = self.get_shared_link(existing_file)
                token = get_token_from_shared_link(shared_link)
                return {'file_name': file_name, 'box_id': existing_file.id, 'token': token}

        try:
            with file_path.open('rb') as file_data:
                uploaded_file = self.client.folder(folder_id).upload_stream(file_data, file_name)
                logging.info(f"Uploaded {file_name} to Box with ID {uploaded_file.id}")
                if force_links:
                    shared_link = self.force_new_shared_link(uploaded_file)
                else:
                    shared_link = self.get_shared_link(uploaded_file)
                token = get_token_from_shared_link(shared_link)
                return {'file_name': file_name, 'box_id': uploaded_file.id, 'token': token}
        except Exception as e:
            logging.error(f"Failed to upload {file_path}: {e}")
            return None

    def file_exists(self, folder_id: str, file_name: str, with_details: bool=False) -> Union[bool, box_object.file.File]:
        """
        Check if a file with the given name exists in the specified folder.

        Args:
            folder_id (str): The ID of the folder to check.
            file_name (str): The name of the file to check.
            with_details (bool, optional): Whether to return the file object with details. Defaults to False.

        Returns:
            bool or boxsdk.object.file.File: True if the file exists, or the file object if with_details is True.
        """
        items = self.client.folder(folder_id).get_items()
        for item in items:
            if item.name == file_name:
                return item if with_details else True
        return False

    def get_shared_link(self, file: box_object.file.File) -> None:
        """
        Ensure that a file has a shared link, creating one if necessary.

        Args:
            file (boxsdk.object.file.File): The file object.

        Returns:
            str: The URL of the shared link.
        """
        # Fetch the file with details about the shared link
        file_with_details = self.client.file(file_id=file.id).get(fields=['shared_link'])
        if file_with_details.shared_link:
            return file_with_details.shared_link['url']
        else:
            shared_link_settings = {
                'shared_link': {
                    'access': 'open',  # or 'company' or 'collaborators', based on your requirement
                    'unshared_at': None  # No expiration date
                }
            }
            file_info = file_with_details.update_info(data=shared_link_settings)
            shared_link = file_info['url']
            return shared_link

    def force_new_shared_link(self, file: box_object.file.File) -> None:
        """
        Ensure that a file has a shared link, creating one if necessary.

        Args:
            file (boxsdk.object.file.File): The file object.

        Returns:
            str: The URL of the shared link.
        """
        # Fetch the file with details about the shared link
        file_with_details = self.client.file(file_id=file.id).get(fields=['shared_link'])
        shared_link_settings = {
            'shared_link': {
                'access': 'open',  # or 'company' or 'collaborators', based on your requirement
                'unshared_at': None  # No expiration date
            }
        }
        file_info = file_with_details.update_info(data=shared_link_settings)
        shared_link = file_info['shared_link']['url']
        return shared_link

    # Does not work. Figure it out? Possibly a ChatGPT hallucination.
    # def check_rate_limit(self):
    #     """
    #     Check the current rate limit of the Box API.
    #     """
    #     try:
    #         # Fetch details of the root folder as an example of a lightweight API call
    #         root_folder = self.client.folder(folder_id=self.root_folder_id).get(fields='id')
    #         if 'X-RateLimit-Remaining' in root_folder.response_headers:
    #             rate_limit_remaining = root_folder.response_headers['X-RateLimit-Remaining']
    #             logging.info(f"Current API Rate Limit Remaining: {rate_limit_remaining}")
    #         else:
    #             logging.warning("Rate limit information is not available in the response headers.")
    #     except Exception as e:
    #         logging.error(f"Error checking rate limit: {e}")


def get_token_from_shared_link(shared_link):
    return shared_link.split('/')[-1]


# Seems to not be needed. Figure it out? Possibly a ChatGPT hallucination.
# def safe_api_call(call: Callable, max_retries: int=5, delay: int=60) -> Union[Any, None]:
#     """
#     Attempt an API call with retries on rate limit errors.

#     Args:
#         call (function): The API call to be made.
#         max_retries (int, optional): The maximum number of retries. Defaults to 5.
#         delay (int, optional): The delay between retries in seconds. Defaults to 60.

#     Returns:
#         Any or None: The result of the API call, or None if all retries failed.
#     """
#     retries = 0
#     while retries < max_retries:
#         try:
#             return call()
#         except BoxAPIException as e:
#             if e.status == 429:  # Check if it's a rate limit error
#                 logging.warning("Rate limit exceeded. Waiting to retry...")
#                 time.sleep(delay * (2 ** retries))  # Exponential backoff
#                 retries += 1
#             else:
#                 logging.error(f"API call failed with error: {e}")
#                 raise e
#         except Exception as e:
#             logging.error(f"Unexpected error: {e}")
#             raise e
#     return None
