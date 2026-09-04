# Allele Extractor
# Copyright (C) 2026  Alona Bokovnia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import argparse
import cv2
import fitz
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from pypdf import PdfReader, PdfWriter, Transformation
import pytesseract
from pytesseract import Output
from thefuzz import fuzz
import time
import datetime
import signal
import multiprocessing as mp
from functools import lru_cache, partial

pytesseract.pytesseract.tesseract_cmd = r".\Tesseract-OCR\tesseract.exe"

LOCUS_LIST = (
    "D3S1358", "D1S1656", "D2S441", "D10S1248", "D13S317", "Penta E",
    "D16S539", "D18S51", "D2S1338", "CSF1PO", "Penta D", "TH01", "vWA",
    "D21S11", "D7S820", "D5S818", "TPOX", "D8S1179", "D12S391", "D19S433",
    "SE33", "D22S1045", "DYS391", "FGA", "DYS576", "DYS570", "DYS389 I",
    "DYS448", "DYS389 II", "DYS19", "DYS391", "DYS481", "DYS549", "DYS533",
    "DYS438", "DYS437", "DYS635", "DYS390", "DYS439", "DYS392", "DYS643",
    "DYS393", "DYS458", "DYS385", "DYS456", "YGATAH4", "B_DYS456",
    "B_DYS389I", "B_DYS390", " B_DYS389II", "G_DYS458", " G_DYS19",
    "G_DYS385", "Y_DYS393", "Y_DYS391", "Y_DYS439", "Y_DYS635", "Y_DYS392",
    "R_Y_GATA_H4", "R_DYS437", "R_DYS438", "R_DYS448"
)

# Fixed set of simplification steps used when reducing contours to polygons.
_APPROX_EPS_STEPS = np.linspace(0.001, 0.05, 10)


@lru_cache(maxsize=64)
def _load_gray(path_str: str):
    """Load an image from disk and convert it to grayscale."""
    try:
        img = cv2.imread(path_str)
    except Exception as e:
        print(f"Error loading image {path_str}: {e}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _rows_with_long_run(mask, min_len):
    """Return the row indices of a 2D boolean array that contain a run of
    at least ``min_len`` True values."""
    if not mask.any():
        return []

    cs = np.cumsum(mask, axis=1, dtype=np.int32)
    # Longest run of True ending at each column, per row.
    reset = np.maximum.accumulate(np.where(mask, 0, cs), axis=1)
    max_run = (cs - reset).max(axis=1)

    # A qualifying row must contain a run of at least ``min_len`` pixels and,
    # like the original loop, at least one "on" pixel (min_len is always > 0
    # here, since it is a fraction of the image width).
    need = max(min_len, 1)
    return [int(y) for y in np.nonzero(max_run >= need)[0]]


class PDFprocessor:
    def __init__(self, file_input: str):
        self.path = Path(file_input)
        self.name = Path(file_input).stem

    def resize_pdf(self, output_pdf: str) -> None:
        """Resizes a PDF to a standard width of 595 points (A4 width) while
        maintaining the aspect ratio. The resized PDF is saved to the 
        specified output path."""
        reader = PdfReader(self.path)
        writer = PdfWriter()

        for page in reader.pages:
            current_width = float(page.mediabox.width)
            current_height = float(page.mediabox.height)

            coef = 595 / current_width
            target_height = coef * current_height

            transformation = Transformation().scale(coef, coef)
            page.add_transformation(transformation)

            page.mediabox.upper_right = (595, target_height)
            writer.add_page(page)

        with open(output_pdf, "wb") as f:
            writer.write(f)

    def split_pdf(self, output_pdf) -> None:
        """Splits a multi-page PDF into individual single-page PDFs and saves
        them in the specified output directory."""
        pdf_to_split = fitz.open(self.path)

        for page_num in range(len(pdf_to_split)):
            new_pdf = fitz.open()

            new_pdf.insert_pdf(
                pdf_to_split, from_page=page_num, to_page=page_num
            )

            output_filename = os.path.join(
                output_pdf,
                f"page{page_num + 1}.pdf"
            )

            new_pdf.save(output_filename)
            new_pdf.close()
        pdf_to_split.close()

    def pdf_to_png(self, output_pdf) -> None:
        """Converts each page of a PDF to a PNG image and saves them in the 
        specified output directory."""
        pdf = fitz.open(self.path)
        for page_num in range(len(pdf)):  # Converts each page of pdf to png
            # (i don't remember why then
            # i need previous def)
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=300)  # dpi for better quality

            img = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )

            output_name = f"{self.name}.png"

            img.save(os.path.join(
                output_pdf, output_name
            ))

        pdf.close()


class ImageProcessor:
    def __init__(self, file_input: str):
        self.path = Path(file_input)
        gray = _load_gray(str(self.path))
        self.image = gray.copy()
        self.original = gray.copy()
        self.name = self.path.stem

    def black_white(self, threshord_arg):
        """Convert the image to binary using a specified threshold value."""
        _, self.image = cv2.threshold(
            self.image, threshord_arg,
            255, cv2.THRESH_BINARY_INV
        )

        return self

    def save_img(self, output_path):
        cv2.imwrite(
            f'{output_path}/{self.path.name}', self.image
        )

    def crop_to_outer_lines(self):
        """Crop the image to the area defined by the largest contour found 
        in the image. This is useful for removing unnecessary borders or 
        whitespace around the main content of the image."""  
        kernel = np.ones((15, 15), np.uint8)

        closed = cv2.morphologyEx(
            self.image, cv2.MORPH_CLOSE,
            kernel, iterations=3
        )

        dilated = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )  # Finding contours

        # Finding the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        self.image = self.original[y:y + h, x:x + w]  # Cropping
        return self

    def crop_color_canals(self, output_png):
        """Crop the image to the area defined by the color canals found 
        in the image."""
        gap_threshold = 300  # Color canals are higher than 300 pixels
        _height, width = self.image.shape

        min_len = width * 0.9  # Lines that are >90% of image width

        # Rows holding a horizontal run of >= min_len "on" pixels
        # (vectorised equivalent of the former per-row scan).
        line_rows = _rows_with_long_run(self.image > 0, min_len)

        # Lines filter
        if line_rows:
            unique_lines = [line_rows[0]]
            for i in range(1, len(line_rows)):
                # Gap more than 5 pixels = different lines
                if line_rows[i] - unique_lines[-1] > 5:
                    unique_lines.append(line_rows[i])

        count = 0

        # Coordinates of the found lines and calculation of
        # the distance between them
        if unique_lines:
            for i in range(len(unique_lines) - 1):
                y_top = unique_lines[i]
                y_bottom = unique_lines[i + 1]
                gap = y_bottom - y_top

                if gap > gap_threshold:
                    cropped_section = self.original[
                        y_top: y_bottom, 0:width
                    ]

                    count += 1
                    cv2.imwrite(
                        f'{output_png}/{self.name}_{count}.png',
                        cropped_section
                    )

            return True

    def locus_coords_function(self):
        """Find the coordinates of loci names in the image using OCR."""
        height_s = self.image.shape[0]
        height_crop = int(height_s * 0.13)
        roi = self.image[:height_crop, :]  # Area of locuses names

        data = pytesseract.image_to_data(
            roi, config='--psm 11 -c load_system_dawg=0 -c load_freq_dawg=0',
            output_type=Output.DICT
        )

        raw_words = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue

            # All text from pytesseract
            raw_words.append({
                'text': text,
                'x1_loc_name': data['left'][i],
                'y1_loc_name': data['top'][i],
                'w_loc_name': data['width'][i],
                'h_loc_name': data['height'][i],
                'x2_loc_name': data['left'][i] + data['width'][i],
                'used': False
            })

        stitched_words = []
        for i, word1 in enumerate(raw_words):
            if word1['used']:
                continue

            current_text = word1['text']
            x1_loc_name, y1_loc_name, x2_loc_name, h_loc_name = \
                word1['x1_loc_name'], \
                word1['y1_loc_name'], \
                word1['x2_loc_name'], \
                word1['h_loc_name']

            word1['used'] = True

            # Word stitching for loci consisting of 2 words
            for j, word2 in enumerate(raw_words):
                if word2['used']:
                    continue
                same_line = abs(
                    word1['y1_loc_name']
                    - word2['y1_loc_name']
                ) < 8

                close_right = 0 <= (
                    word2['x1_loc_name'] - x2_loc_name
                ) < 25

                blacklist = ['(', ')', '[', ']', '{', '}']

                if same_line \
                    and close_right \
                        and not [
                            i for i in blacklist if i in word2['text']
                        ]:

                    current_text = f"{current_text} {word2['text']}"
                    x2_loc_name = word2['x2_loc_name']
                    y1_loc_name = min(y1_loc_name, word2['y1_loc_name'])
                    h_loc_name = max(h_loc_name, word2['h_loc_name'])
                    word2['used'] = True  # For avoiding doubles
                    break

            stitched_words.append({
                'text': current_text,
                'x1_loc_name': x1_loc_name,
                'y1_loc_name': y1_loc_name,
                'w_loc_name': x2_loc_name - x1_loc_name,
                'h_loc_name': h_loc_name
            })

        coord_list = []
        for word in stitched_words:
            detected_text_lower = word['text'].lower()

            best_match_locus = None
            highest_score = 0

            # Search for similarities between words found by
            # pytesseract and words in the list
            for locus_name in LOCUS_LIST:
                target_lower = locus_name.lower()
                similarity = fuzz.ratio(
                    target_lower, detected_text_lower
                )

                # Choosing only what has the greatest similarity
                if similarity > highest_score:
                    highest_score = similarity
                    best_match_locus = locus_name

            if highest_score >= 70 and best_match_locus is not None:
                coord_list.append({
                    'name': best_match_locus,
                    'x1_loc_name': word['x1_loc_name'],
                    'y1_loc_name': word['y1_loc_name'],
                    'w_loc_name': word['w_loc_name'],
                    'h_loc_name': word['h_loc_name']
                })

        return coord_list

    def crop_locus_function(self, locus_coords_name):
        """Find the coordinates of loci borders in the image based on the 
        coordinates of the loci names."""
        width_s = self.image.shape[1]
        height_crop = 0

        for locus in locus_coords_name:
            height_crop = int(
                locus['h_loc_name'] / 2
                + locus['y1_loc_name'] + 30
            )

            break

        roi = self.image[:height_crop, :]  # Area with locus names

        seen = set()
        true_raw_locus_coords = []

        try:
            for r in range(40, 200, 20):
                raw_locus_coords = []

                _, thresh = cv2.threshold(
                    roi, r, 255, cv2.THRESH_BINARY_INV
                )

                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, 30)
                )  # Matrix for vertical lines

                detected_lines = cv2.morphologyEx(
                    thresh, cv2.MORPH_OPEN, kernel
                )  # Transformation of picture so only
                # vertical lines is visible

                lines = cv2.HoughLinesP(
                    detected_lines, 1, np.pi / 180, threshold=20,
                    minLineLength=25, maxLineGap=5
                )  # Detection of vertical lines

                for item in locus_coords_name:
                    x1_loc_name, w_loc_name, h_loc_name = \
                        item['x1_loc_name'], \
                        item['w_loc_name'], \
                        item['y1_loc_name'] \
                        + item['h_loc_name']

                    middle = w_loc_name / 2 + x1_loc_name

                    left_border = 0
                    right_border = width_s

                    # Choosing lines that are closest to locus names
                    if lines is not None:
                        xs = [line[0][0] for line in lines] \
                            + [line[0][2] for line in lines]

                        left_candidates = [x for x in xs if x < (middle)]
                        if left_candidates:
                            left_border = max(left_candidates)
                            if left_border > item['x1_loc_name']:
                                left_border = item['x1_loc_name']

                        right_candidates = [x for x in xs if x > (middle)]

                        if right_candidates:
                            right_border = min(right_candidates)

                            if right_border < item['x1_loc_name'] \
                                    + item['w_loc_name']:

                                right_border = item['x1_loc_name'] \
                                    + item['w_loc_name']

                    borders_dict = {
                        'file_name': self.name,
                        'name': item['name'],
                        'x1_locus': int(left_border),
                        'x2_locus': int(right_border),
                        'h_locus': int(h_loc_name),
                        'middle': middle
                    }

                    raw_locus_coords.append(borders_dict)

                sorted_raw_locus_coords = sorted(
                    raw_locus_coords,
                    key=lambda x: x['x1_locus']
                )

                if not sorted_raw_locus_coords:
                    break

                current_iteration_valid = []

                # Iteration to select the optimal threshold parameter for
                # the binary image so that the loci do not overlap with each
                # other, by taking pairs of loci and comparing the
                # coordinates of their right and left borders.
                for i in range(len(sorted_raw_locus_coords) - 1):
                    cur = sorted_raw_locus_coords[i]
                    nxt = sorted_raw_locus_coords[i + 1]

                    if cur['x2_locus'] >= nxt['x1_locus']:
                        break

                    left_half = cur['middle'] - cur['x1_locus']
                    right_half = cur['x2_locus'] - cur['middle']

                    if (left_half + 100 >= right_half
                            and left_half <= right_half + 100
                            and cur['name'] not in seen):
                        current_iteration_valid.append(cur)
                else:
                    if len(sorted_raw_locus_coords) == 1:
                        break

                    # Comparing the last locus without pair
                    last = sorted_raw_locus_coords[-1]
                    prev = sorted_raw_locus_coords[-2]

                    if last['x1_locus'] >= prev['x2_locus']:
                        left_half = last['middle'] - last['x1_locus']
                        right_half = last['x2_locus'] - last['middle']

                        if (left_half + 100 >= right_half
                                and left_half <= right_half + 100
                                and last['name'] not in seen):
                            current_iteration_valid.append(last)

                    for valid_locus in current_iteration_valid:
                        true_raw_locus_coords.append(valid_locus)
                        seen.add(valid_locus['name'])

                # Every requested locus is resolved - the remaining threshold
                # passes are gated by "name not in seen" and cannot add
                # anything.
                if locus_coords_name and len(seen) >= len(locus_coords_name):
                    break

            return true_raw_locus_coords
        except Exception:
            return []


class AlleleData: 
    def __init__(self, file_input: str):
        self.path = Path(file_input)
        gray = _load_gray(str(self.path))
        self.image = gray.copy()
        self.original = gray.copy()
        self.name = self.path.stem

    def black_white(self, threshord_arg):
        """Convert the image to binary using a specified threshold value."""
        _, self.image = cv2.threshold(
            self.image, threshord_arg,
            255, cv2.THRESH_BINARY
        )

        return self

    def allele_roi_function(self, list_of_borders):
        """Find the region of interest (ROI) in the image where allele data
        is located."""
        min_gap_threshold = 100
        max_gap_threshold = 700
        height, width = self.image.shape

        min_len = width * 0.5

        # Rows holding a horizontal run of >= min_len "off" pixels
        # (vectorised equivalent of the former per-row scan).
        line_rows = _rows_with_long_run(self.image < 1, min_len)

        if line_rows:
            unique_lines = [line_rows[0]]
            for i in range(1, len(line_rows)):
                if line_rows[i] - unique_lines[-1] > 5:
                    unique_lines.append(line_rows[i])

        y_locus_true = None

        if unique_lines:
            for i in range(len(unique_lines) - 1):
                if unique_lines[i + 1] < height - 10:
                    y_top = unique_lines[i]
                    y_bottom = unique_lines[i + 1]
                    gap = y_bottom - y_top

                    if gap > min_gap_threshold \
                            and gap < max_gap_threshold:

                        y_locus_true = y_bottom

        new_list_of_borders = []
        if y_locus_true is not None:
            for border in list_of_borders:
                if border['file_name'] in self.name and y_locus_true:
                    border['y_locus'] = y_locus_true
                    new_list_of_borders.append(border)

        if y_locus_true is None:
            return None

        return new_list_of_borders

    def allele_contours(self, list_of_borders):
        """Find contours in the image that correspond to allele data and 
        extract the data."""
        data_dict = {}
        seen = set()

        for locus in list_of_borders:
            if locus['file_name'] in self.name:
                list_lines = []
                locus_area = self.image[
                    locus['y_locus']:,
                    locus['x1_locus']:locus['x2_locus']
                ]

                h_locus = locus_area.shape[0]

                # Finding all of contours in the image
                contours, _ = cv2.findContours(
                    locus_area, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                boxes = []
                filtered_boxes = []

                for cunt in contours:

                    # Reducing the contour to the smallest number of points
                    # (btw contours here are just a certain number of points)
                    perimeter = cv2.arcLength(cunt, True)
                    for eps in _APPROX_EPS_STEPS:
                        approx = cv2.approxPolyDP(cunt, eps * perimeter, True)

                        if len(approx) < 4:
                            break

                        if len(approx) > 4:
                            continue

                        # If there are 4 points - it's a square
                        if len(approx) == 4:
                            x_bbox, y_bbox, w_bbox, h_bbox = \
                                cv2.boundingRect(approx)

                            # Do not include text with the name of the loci
                            if (h_bbox < 40
                                    or h_bbox > 300
                                    or w_bbox > 300
                                    or h_bbox / 2 > w_bbox
                                    or h_bbox + y_bbox == h_locus):
                                break
                            else:
                                boxes.append(
                                    [x_bbox, y_bbox, w_bbox, h_bbox])
                            break

                # Filtering boxes inside another boxes
                for i, box_a in enumerate(boxes):
                    is_container = False

                    for j, box_b in enumerate(boxes):
                        if i == j:
                            continue

                        if contains(box_a, box_b):
                            is_container = True
                            break

                    if not is_container:
                        filtered_boxes.append(box_a)

                if locus['name'] not in seen:
                    seen.add(locus['name'])

                    # Extraction text from choosed boxes
                    for box in filtered_boxes:
                        allele_roi = self.image[
                            box[1] + locus['y_locus']:box[1] + box[3] +
                            locus['y_locus'], locus['x1_locus'] + box[0]:
                            locus['x1_locus'] + box[0] + box[2]
                        ]

                        data = pytesseract.image_to_string(
                            allele_roi,
                            config='--psm 6 '
                            '-c tessedit_char_whitelist=0123456789.'
                        )

                        changed = False

                        # If there is data in box but the box border touches
                        # the image border, we need to move that box.
                        # For left side
                        if data and box[0] <= 1:
                            left_border, right_border = self.change_border(
                                'left', locus['x1_locus'] + box[0],
                                locus['y_locus'] + box[1], box[2], box[3]
                            )

                            changed = True

                        # For right side
                        right_side_calc = locus['x2_locus'] \
                            - locus['x1_locus'] - box[0] - box[2]

                        if data and right_side_calc <= 1:
                            left_border, right_border = self.change_border(
                                'right', locus['x1_locus'] + box[0],
                                locus['y_locus'] + box[1], box[2], box[3]
                            )

                            changed = True

                        # Calculation of new borders and new data extraction
                        if changed:
                            box[0] = left_border - locus['x1_locus']
                            local_right_border = \
                                right_border - locus['x1_locus']
                            box[2] = local_right_border - box[0]
                            new_allele_roi = self.image[
                                box[1] + locus['y_locus']:
                                box[1] + box[3] + locus['y_locus'],
                                locus['x1_locus'] + box[0]:
                                locus['x1_locus'] + box[0] + box[2]
                            ]

                            data = pytesseract.image_to_string(
                                new_allele_roi, config='--psm 6 '
                                '-c tessedit_char_whitelist=0123456789.'
                            )

                        # Writing text from each box into individual list
                        lines = [line.strip()
                                 for line in data.split() if line.strip()]
                        if lines:

                            # Filtering text and choosing only numbers
                            # (int or float)
                            try:
                                line_int = int(lines[0])
                                if line_int < 50:
                                    list_lines.append(line_int)
                            except Exception:
                                try:
                                    line_flt = float(lines[0])
                                    if line_flt < 50:
                                        list_lines.append(line_flt)
                                except Exception as e:
                                    print(f'Problem with dataframe: {e}')
                                    continue

                    data_dict[locus['name']] = list_lines

        df = pd.DataFrame(
            dict([(k, pd.Series(v))
                  for k, v in data_dict.items()])
        )

        return df

    def change_border(
            self, border, x_change_bbox, y_change_bbox,
            w_change_bbox, h_change_bbox
    ):
        """Change the borders of a bounding box to ensure that the text is
        fully captured within the box."""

        width = self.original.shape[1]

        _, tresh = cv2.threshold(
            self.original, 150, 255, cv2.THRESH_BINARY_INV
        )

        roi_right = tresh[
            y_change_bbox:h_change_bbox
            + y_change_bbox, 0:width
        ]

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 30)
        )

        detected_lines = cv2.morphologyEx(
            roi_right, cv2.MORPH_OPEN, kernel
        )

        lines = cv2.HoughLinesP(
            detected_lines, 5, np.pi / 180, threshold=20,
            minLineLength=40, maxLineGap=5
        )

        left_border = x_change_bbox
        right_border = x_change_bbox + w_change_bbox

        if lines is not None:
            xs = [line[0][0] for line in lines] \
                + [line[0][2] for line in lines]

            if border == 'left':
                left_candidates = [
                    cx for cx in xs
                    if cx < (left_border + w_change_bbox / 2)
                ]

                if left_candidates:
                    left_border = int(max(left_candidates))

            if border == 'right':
                right_candidates = [
                    cx for cx in xs
                    if cx > (right_border - w_change_bbox / 2)
                ]

                if right_candidates:
                    right_border = int(min(right_candidates))

        return left_border, right_border

def contains(box1, box2):
    """Check if box1 contains box2. Each box is defined by a tuple of
    (xmin, ymin, xmax, ymax)."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    return (
        xmin1 < xmin2 and
        ymin1 < ymin2 and
        xmax1 + xmin1 > xmax2 + xmin2 and
        ymax1 + ymin1 > ymax2 + ymin2
    )


def _init_worker():
    """Pool workers ignore Ctrl+C - only the parent process reacts to it,
    then tears the pool down with pool.terminate()."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _process_one_by_pdf(pdf, results_dir):
    """imap-friendly wrapper: derives the per-file temp folder from the 
    PDF."""
    if not str.isascii(pdf.stem):
        newname = pdf.stem.encode('ascii', 'ignore').decode()
    return _process_one(pdf, Path('temp') / newname if 'newname' in locals() 
                        else Path('temp') / pdf.stem, results_dir)


def _process_one(pdf, base_tmp, results_dir):
    """Run the full extraction pipeline for a single PDF.

    Uses an isolated ``base_tmp`` directory so several copies can run in
    parallel worker processes without stepping on each other. Returns a
    tuple ``(status, name, seconds, message)`` where ``status`` is
    ``"success"`` or ``"error"``.
    """
    base_tmp = Path(base_tmp)
    results_dir = Path(results_dir)

    _load_gray.cache_clear()
    shutil.rmtree(base_tmp, ignore_errors=True)

    split_pdf_dir = base_tmp / 'split_pdf'
    img_dir = base_tmp / 'img'
    crop_dir = base_tmp / 'crop'
    for directory in (split_pdf_dir, img_dir, crop_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()  # Set timer

    try:
        try:
            if not str.isascii(pdf.name):
                newname = pdf.name.encode('ascii', 'ignore').decode()
            # Resize into the temp folder instead of overwriting the input PDF.
            resized_pdf = base_tmp / f"resized_{newname if 'newname' 
                                                in locals() else pdf.name}"
            PDFprocessor(pdf).resize_pdf(str(resized_pdf))
            PDFprocessor(resized_pdf).split_pdf(split_pdf_dir)
        except Exception as e:
            print(f'Problem with file: {e}')

        for split_pdf in split_pdf_dir.glob('*.pdf'):
            PDFprocessor(split_pdf).pdf_to_png(img_dir)

        result = False

        for img in img_dir.glob('*.png'):
            try:
                ImageProcessor(img).black_white(150).crop_to_outer_lines()\
                    .save_img(img_dir)
                # crop_to_outer_lines just rewrote this PNG in place.
                _load_gray.cache_clear()

                for r in range(40, 200, 20):
                    result = ImageProcessor(img).black_white(r)\
                        .crop_color_canals(crop_dir)

                    if result is True:
                        break
            except Exception as e:
                print(
                    f'\033[31mProblems with cropping color canals: {e}.\033[0m'
                )

                break

        seen = set()
        list_of_df = []

        for crop in crop_dir.glob('*.png'):
            locus_coords_name = []  # Coordinates loci names
            locus_coords = []  # Coordinates loci borders

            for r in range(40, 200, 20):
                coords = (
                    ImageProcessor(crop).black_white(r)
                    .locus_coords_function()
                )

                if coords:
                    for c in coords:
                        if c['name'] not in seen:
                            seen.add(c['name'])
                            locus_coords_name.append(c)

            locus_coords = (
                ImageProcessor(crop).crop_locus_function(locus_coords_name)
            )

            if locus_coords:
                try:
                    new_locus_coords = None
                    for r in range(50, 190, 20):
                        new_locus_coords = (
                            AlleleData(crop).black_white(r)
                            .allele_roi_function(locus_coords)
                        )

                        if new_locus_coords is None:
                            continue
                        else:
                            break

                    df = (
                        AlleleData(crop).black_white(180)
                        .allele_contours(new_locus_coords)
                    )

                    if df is not None and not df.empty:
                        list_of_df.append(df)

                except Exception as e:
                    print(f'Problem with data extraction from alleles: {e}')
                    continue

        execution_time = time.perf_counter() - start_time

        try:
            full_df = pd.concat(list_of_df, axis=1)
            # Remove duplicate columns
            no_dopplers = full_df.loc[
                :, ~full_df.columns.duplicated()
            ]

            no_dopplers.to_excel(f"{results_dir}/{pdf.name}.xlsx")

            return ("success", pdf.name, execution_time, "")

        except Exception as e:
            print(
                f'\033[31mThere is no dataframe for {pdf.name}.\n'
                f'Error: {e}\033[0m'
            )
            return ("error", pdf.name, execution_time, str(e))
    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)


def main():
    """Main function to handle command-line arguments and orchestrate the
    processing of PDF files."""
    parser = argparse.ArgumentParser(
        description="File processing"
    )

    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to folder with pdf files"
    )

    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of PDF files to process in parallel (default: 1). "
             "Set it up to the number of CPU cores to speed up large batches."
    )

    args = parser.parse_args()
    pdf_dir = args.input
    if not pdf_dir.is_dir():
        parser.error(f"{pdf_dir} is not a folder")

    jobs = max(1, args.jobs)

    # Reading logs
    log_file = Path("processed_files.log")
    if log_file.exists():
        with open(
            log_file, "r", encoding="utf-8"
        ) as f:

            processed_files = set(
                line.split(":")[1].strip() for line in f if (
                    line.startswith("Success:") or line.startswith("Error:")
                )
            )
    else:
        open(log_file, "a", encoding="utf-8")
        processed_files = set()

    all_files = [
        pdf for pdf in pdf_dir.iterdir() if pdf.is_file()
    ]

    count_files = len(all_files)
    processed_count = len(processed_files)

    todo = [
        pdf for pdf in sorted(pdf_dir.glob('*.pdf'))
        if pdf.name not in processed_files
    ]

    def record(result):
        """Record the result of processing a PDF file in the log file."""
        status, name, seconds, _message = result
        now = datetime.datetime.now()
        with open(log_file, "a", encoding="utf-8") as f:
            if status == "success":
                f.write(
                    f"{now}\nSuccess: {name}\n"
                    f"Processing time: {seconds:.2f} seconds\n\n"
                )
            else:
                f.write(f"{now}\nError: {name}\n\n")
        processed_files.add(name)

    interrupted = False

    if jobs == 1:
        # Ctrl+C propagates straight out as KeyboardInterrupt (handled below).
        try:
            for pdf in todo:
                processed_count += 1
                print(
                    f'({processed_count}/{count_files}) '
                    f'Processing file {pdf.name}...'
                )
                result = _process_one(pdf, Path('.\\temp') / pdf.stem, df_dir)
                status, name, seconds, message = result
                if status == "success":
                    print(
                        f'\033[32m{pdf.name} processed in '
                        f'{seconds:.2f} seconds.\033[0m'
                    )
                else:
                    print(
                        f'\033[31m{pdf.name} failed to process.' 
                        f' Error: {message}\033[0m'
                    )

                record(result)

        except KeyboardInterrupt:
            interrupted = True
    else:
        print(
            f'Processing {len(todo)} file(s) with {jobs} worker(s)... '
            '(press Ctrl+C to stop)'
        )

        worker = partial(_process_one_by_pdf, results_dir=df_dir)
        pool = mp.Pool(processes=jobs, initializer=_init_worker)
        try:
            # Poll for results with a short timeout: an unbounded wait can
            # swallow Ctrl+C (notably on Windows), while polling lets the
            # KeyboardInterrupt surface between iterations.
            results = pool.imap_unordered(worker, todo)
            remaining = len(todo)
            while remaining:
                try:
                    result = results.next(timeout=0.5)
                except mp.TimeoutError:
                    continue
                remaining -= 1
                processed_count += 1
                status, name, seconds, _message = result
                if status == "success":
                    print(
                        f'({processed_count}/{count_files}) {name}' 
                        f'[\033[32m{status}\033[0m] in {seconds:.2f} seconds.'
                    )
                else:
                    print(
                        f'({processed_count}/{count_files}) {name}' 
                        f'[\033[31m{status}\033[0m].'
                    )
                record(result)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            interrupted = True
            print('\033[31mStopping - terminating workers...\033[0m')
            pool.terminate()
            pool.join()
        finally:
            pool.terminate()  # harmless if already stopped
            pool.join()

    shutil.rmtree(Path('.\\temp'), ignore_errors=True)

    if interrupted:
        print('\033[31mProcessing was interrupted by the user.\033[0m')
        # Files that were still running are simply not logged, so a re-run
        # picks them up again.


df_dir = Path('.\\results')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            '\033[31mProcessing was interrupted by the user.\033[0m'
        )

        shutil.rmtree(Path('.\\temp'), ignore_errors=True)

    except Exception as e:
        print(f'\033[31mProcessing was interrupted. \nError: {e}\033[0m')

        shutil.rmtree(Path('.\\temp'), ignore_errors=True)
