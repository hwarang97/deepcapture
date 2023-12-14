import io
import os
import random

import requests
from bs4 import BeautifulSoup
from google.cloud import vision
from google.oauth2 import service_account


def download_image(url, destination_path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200 and response.headers["content-type"].startswith(
            "image/"
        ):
            with open(destination_path, "wb") as file:
                file.write(response.content)
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")


def find_image_urls(page_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(page_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            return [img["src"] for img in soup.find_all("img") if "src" in img.attrs]
        else:
            print(f"Failed to load page: {page_url}")
    except requests.RequestException as e:
        print(f"Error loading page {page_url}: {e}")
    return []


def get_random_images(folder_path, number_of_images):
    images = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return random.sample(images, min(number_of_images, len(images)))


def create_augmented_images(folder_path, destination_path, num_images, key_path):
    downloaded = 0
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # 랜덤으로 이미지 선택
    selected_images = get_random_images(folder_path, num_images)  # 5개의 이미지를 랜덤으로 선택

    for image_name in selected_images:
        file_path = os.path.join(folder_path, image_name)

        # 이미지 파일 읽기
        with io.open(file_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        # 유사한 이미지 검색 요청
        response = client.web_detection(image=image)
        web_detection = response.web_detection

        # 페이지 URL 추출 및 이미지 다운로드
        if web_detection.pages_with_matching_images:
            for page in web_detection.pages_with_matching_images:
                image_urls = find_image_urls(page.url)
                for img_url in image_urls:
                    if img_url.startswith("http"):
                        download_image(
                            img_url, f"{destination_path}{downloaded + 1}.jpg"
                        )
                        downloaded += 1

    print(downloaded)
    print("Image processing completed.")
