from typing import Any

import cv2
import pytesseract
import os
import re

from cv2 import Mat
from numpy import ndarray, dtype

custom_config = r'-l grc+eng --psm 1'

cropped_size_dict = {
    'dataset': 890,
    'dataset2': 900
}


def show_img(img):
    cv2.imwrite('color_img.jpg', img)
    cv2.imshow("image", img)
    cv2.waitKey()


def prepare_img(img, crop_coefficient=0, debug=0):
    img = img[crop_coefficient:, :]

    if debug == 1:
        show_img(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

    if debug == 1:
        show_img(thresh1)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def recognize_and_crop(contours, img, debug=0):
    im2 = img.copy()
    height = im2.shape[0]
    width = im2.shape[1]
    recognized_text = ""
    cropped = im2

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if int(0.05 * height) <= h and w >= int(1.0 * width):
            cropped = im2[y:y + h, x:x + int(w / 5 * 4)]
            if cropped.shape[0] > 120:
                cropped = cropped[0: int(cropped.shape[0] / 2), :]

            if debug == 1:
                show_img(cropped)
            text = pytesseract.image_to_string(cropped, lang='eng+grc+ell').replace("\n", "")

            cropped = im2[y:y + h, x:x + w]
            if cropped.shape[0] > 120:
                cropped_test = cropped[20: 70, x + int(w / 7 * 6):x + w]
            else:
                cropped_test = cropped[:, x + int(w / 7 * 6):x + w]

            if debug == 1:
                show_img(cropped_test)

            text_test = pytesseract.image_to_string(cropped_test, lang='eng+grc').replace("\n", "")
            if text != "":
                recognized_text += text

            text_test = text_test.replace("yu", "u")

            if text_test not in text:
                recognized_text += text_test

    if debug == 1:
        print(recognized_text)

    scale_x = re.findall(r'x\d+.\d{0,3}', recognized_text)[0]

    recognized_text = recognized_text[-10:]

    length_nm = re.findall(r'\d+.\d{0,3}nm', recognized_text)
    if len(length_nm) == 0:
        length_um: list[str] = re.findall(r'\d+.\d{0,3}[up]?m', recognized_text)
        parsed_length_um = ''.join(filter(lambda item: item.isdigit() or item == '.', length_um[0]))
        result_size = float(parsed_length_um) * 1000
    else:
        result_size = float(''.join(filter(lambda item: item.isdigit(), length_nm[0])))

    return scale_x, int(result_size), cropped


def get_nm_of_pixel(scale_x, length_nm, cropped_image: Mat | ndarray[Any, dtype] | ndarray):
    left = 0
    right = 0

    meet_zero = False

    for index, pixel in enumerate(cropped_image[0]):
        if not left and pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            left = index

        if left and not meet_zero and not pixel[0] and not pixel[1] and not pixel[2]:
            meet_zero = True

        if left and meet_zero and pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            right = index
            break

    # for index, pixel in enumerate(reversed(cropped_image[0])):
    #     if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
    #         right = len(cropped_image[0]) - index
    #         break

    different_indexes = right - left

    assert  different_indexes >= 0

    return round(length_nm / different_indexes, 3)


def main():
    debug = 0

    dirname = 'dataset2'
    path_to_dirname = f"./{dirname}/"
    directory = os.fsencode(path_to_dirname)

    for file in os.listdir(directory):
        filename = os.fsdecode(directory + file)
        img = cv2.imread(filename)

        contours = prepare_img(img, cropped_size_dict[dirname], debug)

        scale_x, length_nm, cropped_image = recognize_and_crop(contours, img[cropped_size_dict[dirname]:, :], debug)
        pixel_nm = get_nm_of_pixel(scale_x, length_nm, cropped_image[10: , :])

        print(file, scale_x, length_nm, pixel_nm, end='\n')


if __name__ == "__main__":
    main()
