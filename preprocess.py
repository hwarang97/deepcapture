import os
import random

from PIL import Image
from PIL import ImageSequence


def create_augmented_images(source_folder, target_folder, num_images):
    """이미지 데이터 증강 함수"""
    image_paths = [
        os.path.join(source_folder, file)
        for file in os.listdir(source_folder)
        if file.endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

    for _ in range(num_images):
        try:
            img1_path, img2_path = random.sample(image_paths, 2)

            img1, img2 = open_image(img1_path), open_image(img2_path)
            if img1 is None or img2 is None:
                continue

            new_image = augment_images(img1, img2)
            if new_image:
                save_new_image(new_image, target_folder)

        except Exception as e:
            print(f"Skipping image pair due to error: {e}")


def open_image(image_path):
    """이미지를 안전하게 여는 함수"""
    try:
        img = Image.open(image_path)
        if image_path.endswith(".gif"):
            img = next(ImageSequence.Iterator(img))  # 첫 번째 프레임만 사용
        return img
    except IOError:
        print(f"Error opening image {image_path}. Skipping...")
        return None


def augment_images(img1, img2):
    """이미지 증강 함수"""
    try:
        min_width = min(img1.width, img2.width)
        min_height = min(img1.height, img2.height)
        img1 = img1.resize((min_width, min_height))
        img2 = img2.resize((min_width, min_height))

        left_half = img1.crop((0, 0, min_width // 2, min_height))
        right_half = img2.crop((min_width // 2, 0, min_width, min_height))

        new_image = Image.new("RGB", (min_width, min_height))
        new_image.paste(left_half, (0, 0))
        new_image.paste(right_half, (min_width // 2, 0))
        return new_image
    except Exception:
        print("Error augmenting images. Skipping...")
        return None


def save_new_image(new_image, target_folder):
    """이미지 저장 함수"""
    new_image_path = os.path.join(
        target_folder, f"augmented_{random.randint(1, 10000)}.jpg"
    )
    new_image.save(new_image_path)


# 폴더 경로를 설정합니다
source_folder = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/term project/dredze-spam_archive/spam_archive_jmlr"
target_folder = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/term project/new_dredze_spam_archive"

# 아직 실제 파일 시스템에 접근할 수 없으므로, 실제 환경에서 이 코드를 실행해야 합니다.
# 여기서는 함수 정의만 제공하며, 원하는 폴더에 적용하실 수 있습니다.

# 예시: create_augmented_images(source_folder, target_folder, 10) - 10개의 새로운 이미지 생성
create_augmented_images(source_folder, target_folder, 300)
