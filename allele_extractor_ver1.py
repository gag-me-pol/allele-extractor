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

LOCUS_LIST = (
    "D3S1358", "D1S1656", "D2S441", "D10S1248", "D13S317", "Penta E", "D16S539", 
    "D18S51", "D2S1338", "CSF1PO", "Penta D", "TH01", "vWA", "D21S11", "D7S820", 
    "D5S818", "TPOX", "D8S1179", "D12S391", "D19S433", "SE33", "D22S1045", "DYS391", 
    "FGA", "DYS576", "DYS570", "DYS389 I", "DYS448", "DYS389 II", "DYS19", "DYS391", 
    "DYS481", "DYS549", "DYS533", "DYS438", "DYS437", "DYS635", "DYS390", "DYS439", 
    "DYS392", "DYS643", "DYS393", "DYS458", "DYS385", "DYS456", "YGATAH4", "B_DYS456", 
    "B_DYS389I", "B_DYS390", " B_DYS389II", "G_DYS458", " G_DYS19", "G_DYS385", 
    "Y_DYS393", "Y_DYS391", "Y_DYS439", "Y_DYS635", "Y_DYS392", "R_Y_GATA_H4", 
    "R_DYS437", "R_DYS438", "R_DYS448"
)

# Processes PDF to PNG
class PDFprocessor:
    def __init__(self, file_input: str):
        self.path = Path(file_input)
        self.name = Path(file_input).stem

    # All pdfs need to be in one size (at least approximately)
    def resize_pdf(self, output_pdf: str) -> None:
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

    # Splitting pdf pages into separate pdfs 
    def split_pdf(self, output_pdf) -> None:
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

    # Convertation to png for opencv and pytesseract
    def pdf_to_png(self, output_pdf) -> None:
        pdf = fitz.open(self.path)
        for page_num in range(len(pdf)): # Converts each page of pdf to png 
                                         # (i don't remember why then 
                                         # i need previous def)
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=300) # dpi for better quality

            img = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
                ) 
            
            output_name = f"{self.name}.png"

            img.save(os.path.join(
                output_pdf, output_name
                ))
            
        pdf.close()

# Processes PNG 
class ImageProcessor:
    def __init__(self, file_input: str):
        self.path = Path(file_input)
        self.image = cv2.imread(file_input)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.original = self.image.copy()
        self.name = Path(file_input).stem

    # Converts grayscale image into binary by comparing each pixel 
    # intensity against a specified threshold value
    def black_white(self, threshord_arg):
        _, self.image = cv2.threshold(
            self.image, threshord_arg, 
            255, cv2.THRESH_BINARY_INV
            )
        
        return self
    
    def save_img(self, output_path):
        cv2.imwrite(
            f'{output_path}/{self.path.name}', self.image
            )

    # Cropping outer border (it can be different in size, 
    # so easier just cut it off)
    def crop_to_outer_lines(self):
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
            ) # Finding contours

        largest_contour = max(contours, key=cv2.contourArea) # Finding the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        self.image = self.original[y:y+h, x:x+w] # Cropping
        return self

    # Cutting color canals
    def crop_color_canals(self, output_png):
        gap_threshold = 300 # Color canals are higher than 300 pixels
        height, width = self.image.shape
        
        min_len = width * 0.9 # Lines that are >90% of image width  
        line_rows = []

        # Searching by every pixel in y-coordinate
        for y in range(height): 
            row = self.image[y, :]
            on_pixels = np.where(row > 0)[0]
            if len(on_pixels) == 0: continue

            diffs = np.diff(on_pixels)
            splits = np.where(diffs > 1)[0] + 1
            segments = np.split(on_pixels, splits)

            for seg in segments:
                if min_len <= len(seg):
                    line_rows.append(y)
                    break

        # Lines filter 
        if line_rows:
            unique_lines = [line_rows[0]]
            for i in range(1, len(line_rows)):
                if line_rows[i] - unique_lines[-1] > 5: # Gap more than 5 pixels = different lines
                    unique_lines.append(line_rows[i])

        count = 0

        # Coordinates of the found lines and calculation of 
        # the distance between them
        if unique_lines:
            for i in range(len(unique_lines) - 1):
                y_top = unique_lines[i]
                y_bottom = unique_lines[i+1]
                gap = y_bottom - y_top

                if gap > gap_threshold:
                    cropped_section = self.original[
                        y_top : y_bottom, 0:width
                        ] 
                    
                    count += 1
                    cv2.imwrite(
                        f'{output_png}/{self.name}_{count}.png', 
                        cropped_section
                        )
                    
            return True

    def locus_coords_function(self):          
        height_s = self.image.shape[0]
        height_crop = int(height_s*0.13)
        roi = self.image[:height_crop, :] # Area of locuses names

        data = pytesseract.image_to_data(
            roi, config = '--psm 11 -c load_system_dawg=0 -c load_freq_dawg=0', 
            output_type=Output.DICT
            )
        
        raw_words = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text: continue

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
            if word1['used']: continue
            
            current_text = word1['text']
            x1_loc_name, y1_loc_name, x2_loc_name, h_loc_name = \
                word1['x1_loc_name'], \
                    word1['y1_loc_name'], \
                        word1['x2_loc_name'], \
                            word1['h_loc_name']
            
            word1['used'] = True
            
            # Word stitching for loci consisting of 2 words
            for j, word2 in enumerate(raw_words):
                if word2['used']: continue
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
                    word2['used'] = True # For avoiding doubles
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
            for l in LOCUS_LIST:
                target_lower = l.lower()
                similarity = fuzz.ratio(
                    target_lower, detected_text_lower
                    )
                
                # Choosing only what has the greatest similarity
                if similarity > highest_score:
                    highest_score = similarity
                    best_match_locus = l
            
            if highest_score >= 70 and best_match_locus is not None:
                coord_list.append({
                    'name': best_match_locus,
                    'x1_loc_name': word['x1_loc_name'],
                    'y1_loc_name': word['y1_loc_name'],
                    'w_loc_name': word['w_loc_name'],
                    'h_loc_name': word['h_loc_name']
                })

        return coord_list

    # Finding coordinates of loci borders
    def crop_locus_function(self, locus_coords_name):
        width_s = self.image.shape[1]
        height_crop = 0

        for locus in locus_coords_name:
            height_crop = int(
                locus['h_loc_name'] / 2 
                + locus['y1_loc_name'] + 30
                )
            
            break

        roi = self.image[:height_crop, :] # Area with locus names
        
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
                    ) # Matrix for vertical lines

                detected_lines = cv2.morphologyEx(
                    thresh, cv2.MORPH_OPEN, kernel
                    ) # Transformation of picture so only 
                      # vertical lines is visible

                lines = cv2.HoughLinesP(
                    detected_lines, 1, np.pi/180, threshold=20, 
                    minLineLength=25, maxLineGap=5
                    ) # Detection of vertical lines 
                
                for item in locus_coords_name:
                    x1_loc_name, w_loc_name, h_loc_name = \
                        item['x1_loc_name'], \
                            item['w_loc_name'], \
                                item['y1_loc_name'] \
                                    + item['h_loc_name']

                    middle = w_loc_name/2 + x1_loc_name

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
                # the binary image so that the loci 
                # Do not overlap with each other by taking pairs of loci 
                # and compare their coordinates of right and left borders
                for i in range(len(sorted_raw_locus_coords) - 1):
                    if sorted_raw_locus_coords[i]['x2_locus'] \
                        >= sorted_raw_locus_coords[i+1]['x1_locus']:
                        break
                    else:
                        if sorted_raw_locus_coords[i]['middle'] \
                            - sorted_raw_locus_coords[i]['x1_locus'] \
                                + 100 \
                                    >= sorted_raw_locus_coords[i]['x2_locus'] \
                                        - sorted_raw_locus_coords[i]['middle']:
                            
                            if sorted_raw_locus_coords[i]['middle'] \
                                - sorted_raw_locus_coords[i]['x1_locus'] \
                                    <= sorted_raw_locus_coords[i]['x2_locus']\
                                        - sorted_raw_locus_coords[i]['middle']\
                                            + 100:
                                
                                if sorted_raw_locus_coords[i]['name'] not in seen:
                                    current_iteration_valid.append(sorted_raw_locus_coords[i])
                else:
                    if len(sorted_raw_locus_coords) == 1:
                        break

                    # Comparing the last locus without pair
                    if sorted_raw_locus_coords[-1]['x1_locus'] \
                        >= sorted_raw_locus_coords[-2]['x2_locus']:

                        if sorted_raw_locus_coords[-1]['middle'] \
                            - sorted_raw_locus_coords[-1]['x1_locus'] \
                                + 100 \
                                    >= sorted_raw_locus_coords[-1]['x2_locus'] \
                                        - sorted_raw_locus_coords[-1]['middle']:
                            
                            if sorted_raw_locus_coords[-1]['middle'] \
                                - sorted_raw_locus_coords[-1]['x1_locus'] \
                                    <= sorted_raw_locus_coords[-1]['x2_locus'] \
                                        - sorted_raw_locus_coords[-1]['middle'] \
                                            + 100:
                                
                                if sorted_raw_locus_coords[-1]['name'] not in seen:
                                    current_iteration_valid.append(
                                        sorted_raw_locus_coords[-1]
                                        )
                    
                    for valid_locus in current_iteration_valid:
                        true_raw_locus_coords.append(valid_locus)
                        seen.add(valid_locus['name'])

            return true_raw_locus_coords
        except:
            return []

# Extracting data of alleles
class AlleleData: # Input - color canal

    def __init__(self, file_input: str):
        self.path = Path(file_input)
        self.image = cv2.imread(file_input)

        self.image = cv2.cvtColor(
            self.image, cv2.COLOR_BGR2GRAY
            )
        
        self.original = self.image.copy()
        self.name = Path(file_input).stem

    def black_white(self, threshord_arg):
        _, self.image = cv2.threshold(
            self.image, threshord_arg, 
            255, cv2.THRESH_BINARY
            )
        
        return self

    # Finding area where are allele data    
    def allele_roi_function(self, list_of_borders):
        min_gap_threshold = 100
        max_gap_threshold = 700
        height, width = self.image.shape
        
        min_len = width * 0.5
        line_rows = []

        # Searching by every pixel in y-coordinate 
        # (the same as for color canals)
        for y in range(height): 
            row = self.image[y, :]
            on_pixels = np.where(row < 1)[0]
            if len(on_pixels) == 0: continue

            diffs = np.diff(on_pixels)
            splits = np.where(diffs > 1)[0] + 1
            segments = np.split(on_pixels, splits)

            for seg in segments:
                if min_len <= len(seg):
                    line_rows.append(y)
                    continue

        if line_rows:
            unique_lines = [line_rows[0]]
            for i in range(1, len(line_rows)):
                if line_rows[i] - unique_lines[-1] > 5: 
                    unique_lines.append(line_rows[i])

        y_locus_true = None

        if unique_lines:
            for i in range(len(unique_lines) - 1):
                if unique_lines[i+1] < height-10:
                    y_top = unique_lines[i]
                    y_bottom = unique_lines[i+1]
                    gap = y_bottom - y_top

                    if gap > min_gap_threshold \
                        and gap < max_gap_threshold:

                        y_locus_true = y_bottom

        new_list_of_borders = []
        if y_locus_true is not None:
            for l in list_of_borders:
                if l['file_name'] in self.name \
                    and y_locus_true:

                    l['y_locus'] = y_locus_true
                    new_list_of_borders.append(l) 

        if y_locus_true is None:
            return None

        return new_list_of_borders

    # Finding boxes around allele data
    def allele_contours(self, list_of_borders):
        data_dict = {}
        seen = set()

        for l in list_of_borders:
            if l['file_name'] in self.name:
                list_lines = []
                locus_area = self.image[
                    l['y_locus']:, 
                    l['x1_locus']:l['x2_locus']
                    ]

                h_locus = locus_area.shape[0]

                # Finding all of contours in the image
                contours, _ = cv2.findContours(
                    locus_area, cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                    ) 

                boxes = []
                filtered_boxes = []  

                for _, cunt in enumerate(contours):

                    # Reducing the contour to the smallest number of points
                    # (btw contours here are just a certain number of points)
                    for eps in np.linspace(0.001, 0.05, 10): 
                        perimeter = cv2.arcLength(cunt, True) 
                        approx = cv2.approxPolyDP(cunt, eps*perimeter, True)

                        if len(approx) < 4:
                            break
                        
                        if len(approx) > 4:
                            continue
                         
                        if len(approx) == 4: # If there's 4 points - it's a square
                            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(approx)

                            # Do not include text with the name of the loci
                            if h_bbox<40 \
                                or h_bbox>300 \
                                    or w_bbox>300 \
                                        or h_bbox/2>w_bbox \
                                            or h_bbox+y_bbox == h_locus: 
                                break
                            else:
                                boxes.append([x_bbox, y_bbox, w_bbox, h_bbox])
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

                if l['name'] not in seen:
                    seen.add(l['name'])

                    # Extraction text from choosed boxes 
                    for box in filtered_boxes:
                        allele_roi = self.image[
                            box[1]+l['y_locus']:box[1]+box[3]+ 
                            l['y_locus'], l['x1_locus']+box[0]: \
                            l['x1_locus']+box[0]+box[2]
                            ]
                        
                        data = pytesseract.image_to_string(
                            allele_roi, config = 
                            '--psm 6 -c tessedit_char_whitelist=0123456789.'
                            )
                        
                        changed = False 

                        # If there is data in box but border of box is touching border 
                        # of image - we need to change coordinates of that box
                        # For left side
                        if data and box[0] <= 1:
                            left_border, right_border = self.change_border(
                                'left', l['x1_locus']+box[0], 
                                l['y_locus']+box[1], box[2], box[3]
                                )
                            
                            changed = True

                        # For right side
                        right_side_calc = l['x2_locus']-l['x1_locus'] \
                            -box[0]-box[2]

                        if data and right_side_calc <= 1:
                            left_border, right_border = self.change_border(
                                'right', l['x1_locus']+box[0], 
                                l['y_locus']+box[1], box[2], box[3]
                                )
                            
                            changed = True

                        # Calculation of new borders and new data extraction
                        if changed == True:
                            box[0] = left_border - l['x1_locus']
                            local_right_border = right_border - l['x1_locus']
                            box[2] = local_right_border - box[0]
                            new_allele_roi = self.image[
                                box[1]+l['y_locus'] : box[1]+box[3]+l['y_locus'], 
                                l['x1_locus']+box[0] : l['x1_locus']+box[0]+box[2]
                                ]
                            
                            data = pytesseract.image_to_string(
                                new_allele_roi, config = '--psm 6 '
                                '-c tessedit_char_whitelist=0123456789.'
                                )

                        # Writing text from each box into individual list
                        lines = [line.strip() for line in data.split() if line.strip()] 
                        if lines:
                            
                            # Filtering text and choosing only numbers (int or float)
                            try:
                                line_int = int(lines[0])
                                list_lines.append(line_int)
                            except:
                                try:
                                    line_flt = float(lines[0])
                                    list_lines.append(line_flt)
                                except Exception as e:
                                    print(f'Problem with dataframe: {e}')
                                    continue
                                        
                    data_dict[l['name']] = list_lines
        
        df = pd.DataFrame(
            dict([(k, pd.Series(v)) 
                  for k, v in data_dict.items()])
                )
        
        return df

    # Function for changing borders
    def change_border(
            self, border, x_change_bbox, y_change_bbox, 
            w_change_bbox, h_change_bbox
            ):
        
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
            detected_lines, 5, np.pi/180, threshold=20,
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
                    if cx < (left_border + w_change_bbox/2)
                    ]
                
                if left_candidates:
                    left_border = int(max(left_candidates))
                        
            if border == 'right':
                right_candidates = [
                    cx for cx in xs 
                    if cx > (right_border - w_change_bbox/2)
                    ]
                
                if right_candidates:
                    right_border = int(min(right_candidates))

        return left_border, right_border

# Function for checking if one box contains another one
def contains(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    return (
        xmin1 < xmin2 and 
        ymin1 < ymin2 and 
        xmax1+xmin1 > xmax2+xmin2 and 
        ymax1+ymin1 > ymax2+ymin2
        )

def main():
    # Arguments for input files
    parser = argparse.ArgumentParser(
        description="File processing"
        )
    
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to folder with pdf files"
        )

    args = parser.parse_args()
    pdf_dir = args.input
    if not pdf_dir.is_dir():
        parser.error(f"{pdf_dir} is not a folder")

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
    print(processed_files)
    all_files = [
        pdf for pdf in pdf_dir.iterdir() if pdf.is_file()
        ]
    
    count_files = len(all_files)
    processed_count = len(processed_files)

    for pdf in pdf_dir.glob('*.pdf'):

        if pdf.name in processed_files:
            continue

        processed_count += 1
        print(
            f'({processed_count}/{count_files}) Processing file {pdf.name}...'
            )

        start_time = time.perf_counter() # Set timer
        df_dir = Path('df_dir')

        for directory in [split_pdf_dir, img_dir, crop_dir, df_dir]:
            Path(directory).mkdir(exist_ok=True)

        try:
            PDFprocessor(pdf).resize_pdf(f"{pdf_dir}/{pdf.name}")
            PDFprocessor(pdf).split_pdf(split_pdf_dir)

        except Exception as e:
            print(f'Problem with file: {e}')

        for split_pdf in split_pdf_dir.glob('*.pdf'):
            PDFprocessor(split_pdf).pdf_to_png(img_dir)

        result = False

        for img in img_dir.glob('*.png'):
            try:
                ImageProcessor(img).black_white(150).crop_to_outer_lines()\
                    .save_img(img_dir)
                
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
            locus_coords_name = [] # Coordinates loci names
            locus_coords = [] # Coordinates loci borders 

            for r in range(40, 200, 20):
                coords = ImageProcessor(crop).black_white(r).locus_coords_function()    
 
                if coords:
                    for c in coords:
                        if c['name'] not in seen:
                            seen.add(c['name'])
                            locus_coords_name.append(c)

            locus_coords = ImageProcessor(crop).crop_locus_function(locus_coords_name)

            if locus_coords:
                try:
                    new_locus_coords = None
                    for r in range (50, 190, 20):
                        new_locus_coords = AlleleData(crop).black_white(r)\
                            .allele_roi_function(locus_coords)
                        
                        if new_locus_coords is None:
                            continue
                        else:
                            break

                    df = AlleleData(crop).black_white(180)\
                        .allele_contours(new_locus_coords)

                    if df is not None and not df.empty:
                        list_of_df.append(df)

                except Exception as e:
                    print(f'Problem with data extraction from alleles: {e}')
                    continue
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        try:
            full_df = pd.concat(list_of_df, axis=1)
            # Remove duplicate columns
            no_dopplers = full_df.loc[
                :, ~full_df.columns.duplicated()
                ]
             
            no_dopplers.to_excel(f"{df_dir}/{pdf.name}.xlsx")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()}\nSuccess: {pdf.name}\nProcessing time: {execution_time:.2f} seconds\n\n")

            processed_files.add(pdf.name)

            print(
                f'\033[32mFile {pdf.name} was processed in {execution_time:.2f} seconds.\033[0m'
                )

        except Exception as e:
            print(
                f'\033[31mThere is no dataframe for {pdf.name}.\nError: {e}\033[0m'
                )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()}\nError: {pdf.name}\n\n")

        shutil.rmtree(crop_dir)
        shutil.rmtree(split_pdf_dir)
        shutil.rmtree(img_dir)

split_pdf_dir = Path('split_pdf_dir')
img_dir = Path('img_dir')
crop_dir = Path('crop_dir')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            '\033[31mProcessing was interrupted by the user.\033[0m'
            )
        
        shutil.rmtree(crop_dir)
        shutil.rmtree(split_pdf_dir)
        shutil.rmtree(img_dir)
    except Exception as e:
        print(f'\033[31mProcessing was interrupted. \nError: {e}\033[0m')
        shutil.rmtree(crop_dir)
        shutil.rmtree(split_pdf_dir)
        shutil.rmtree(img_dir)
