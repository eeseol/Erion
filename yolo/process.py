# -*- coding: utf-8 -*-
import os

# 경로 설정
test_image_dir = 'test/images'
train_image_dir = 'train/images'
test_txt_path = 'data/obj/test.txt'
train_txt_path = 'data/obj/train.txt'

def create_image_list(image_dir, output_file):
    with open(output_file, 'w') as file:
        for image_name in os.listdir(image_dir):
            if image_name.endswith('.jpg'):
                title = os.path.splitext(image_name)[0]
                file.write("{}/{}.jpg\n".format(image_dir, title))

# test.txt 파일 생성
create_image_list(test_image_dir, test_txt_path)

# train.txt 파일 생성
create_image_list(train_image_dir, train_txt_path)

print("파일 생성이 완료되었습니다.")

