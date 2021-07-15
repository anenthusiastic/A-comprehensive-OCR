import cv2
import pytesseract

from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox

from tkinter import ttk

from PIL import ImageTk
from PIL import Image as PIL_Image
import tempfile
import os

import numpy as np
import imutils
from imutils import contours
from matplotlib import pyplot as plt



window = Tk()
window.title('Image to Text Application')
window.geometry('1200x600')

tessdata_dir_config = '--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

image = PIL_Image.open("background.jpg")
background_image = image.resize((1200,600), PIL_Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(background_image)
background_image_label= Label(window,image=background_image)
background_image_label.place(x=0, y=0, relwidth=1, relheight=1)
image.close()

image_overview= ""
filepath = ""
is_new_image = False
temp_filename = ""
output = ""

# define a dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

ocr_A_reference = "ocr-a.png"


# load the reference OCR-A image from disk
ref = cv2.imread(ocr_A_reference)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

for (i, c) in enumerate(refCnts):
    # compute the bounding box for the digit, extract it, and resize
    # it to a fixed size
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # update the digits dictionary, mapping the digit name to the ROI
    digits[i] = roi

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def load_image():
    global filepath
    filepath = filedialog.askopenfilename(filetypes = (("jpeg files","*.jpeg"),("jpg files","*.jpg"),
                                                       ("png files","*.png"),("tiff files","*.tiff"),("all files","*.*")))
    if(filepath is ""):
        return

    """
    pre, ext = os.path.splitext(filepath)
    if(not ext.islower()):
        os.rename(filepath, pre + ext.lower())
        filepath = pre + ext.lower()
    """

    try:
        global image
        image = PIL_Image.open(filepath)
        global image_overview
        image_overview = image.resize((500,350), PIL_Image.ANTIALIAS)
        image_overview = ImageTk.PhotoImage(image_overview)
        image.close()
        global is_new_image
        is_new_image = True
        if(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
    except:
        messagebox.showerror("Type Error","This document type is not supported!")
        return

    image_label = Label(window, text='Overview of the Original Image', font="Verdana 13 bold", fg='blue')
    image_label.place(x=80, y=50)

    image_overview_label = Label(window, image=image_overview)
    image_overview_label.place(x=50, y=100)

    operation_combobox.place(x=50, y=500)

    #messagebox.showinfo("Successfull","File loaded successfully")


load_image_btn = Button(window, text = "Load Document",font="Times, 10",bg='light blue',
                        command=load_image)
load_image_btn.place(x=80, y=550)

def search_text_in_document(search_text,img):
    if (not search_text.strip()):
        messagebox.showerror("Blank", "Don't left blank the search entry!")
        return

    search_text_words = search_text.split(" ")
    stw_len = len(search_text_words)
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng')
    except TypeError:
        messagebox.showerror("Unsupported", "Unsupported image object!")
        return

    n_boxes = len(d['level'])
    search_image = cv2.imread(temp_filename)
    overlay = search_image.copy()
    flag = False

    for i in range(n_boxes - stw_len + 1):
        text_word = d['text'][i].lower()
        if (text_word == search_text_words[0]):
            # cv2.rectangle(img, (x, y), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            if (stw_len == 1):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(overlay, (x, y),
                              (x + w,
                               y + h),
                              (255, 0, 0), -1)
                flag = True
                continue

            line_jump_indexes = [i]
            if (not d['text'][i + 1].strip()):
                line_jump_indexes.append(i)

            line_jump = 0
            j = 1
            while (j < len(search_text_words) + line_jump and (i + j < n_boxes)):
                if (d['text'][i + j].lower() == search_text_words[j - line_jump]):
                    if (i + j + 1 < n_boxes and (not d['text'][i + j + 1].strip())):
                        line_jump_indexes.append(i + j)

                elif (not d['text'][i + j].strip()):
                    line_jump += 1
                    if (i + j + 1 < n_boxes and d['text'][i + j + 1].strip()):
                        line_jump_indexes.append(i + j + 1)
                else:
                    break

                if (j == len(search_text_words) - 1 + line_jump):
                    line_jump_indexes.append(i + j)
                    flag = True

                    lji_len = len(line_jump_indexes)
                    if (lji_len % 2 == 1):
                        print(line_jump_indexes.pop())

                    half = int(lji_len / 2)
                    for i in range(half):
                        x, y, w, h = (d['left'][line_jump_indexes[i * 2]], d['top'][line_jump_indexes[i * 2]],
                                      d['left'][line_jump_indexes[i * 2 + 1]] + d['width'][
                                          line_jump_indexes[i * 2 + 1]],
                                      d['top'][line_jump_indexes[i * 2 + 1]] + d['height'][
                                          line_jump_indexes[i * 2 + 1]])
                        cv2.rectangle(overlay, (x, y), (w, h), (255, 0, 0), -1)
                j += 1

    if (flag):

        alpha = 0.4  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        img_new = cv2.addWeighted(overlay, alpha, search_image, 1 - alpha, 0)
        r = 1000.0 / img_new.shape[1]  # resizing image without loosing aspect ratio
        dim = (1000, int(img_new.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Search result', resized)
        cv2.waitKey(0)
        save_image(resized)

    else:
        messagebox.showinfo("No Match", "Searched text didn't found!")


def save_image(image):
    result = messagebox.askquestion('SAVE Image', 'Do you want to save the image?')
    if result == 'yes':
        pre, ext = os.path.splitext(filepath)
        output_filepath = filedialog.asksaveasfilename(initialfile = "search_output_"+pre.split("/")[-1])
        if output_filepath.strip():
            
            cv2.imwrite(output_filepath+"."+ext, image)
            messagebox.showinfo('SAVED', 'Saving was completed successfully')
        else:
            messagebox.showerror('Empty file name', 'Filename can not be empty!')
    
def set_image_dpi_resize():
    img = PIL_Image.open(filepath)
    length_x, width_y = img.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = img.resize(size, PIL_Image.ANTIALIAS)
    pre, ext = os.path.splitext(filepath)
    suffix = "."+ext
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_filename = temp_file.name
    image_resize.save(temp_filename,dpi = (300,300))
    img.close()
    return temp_filename

def set_image_dpi():
    image = PIL_Image.open(filepath)
    image_resize = image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename

def remove_shadows(img):

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def correct_skewness(img):

    gray = cv2.bitwise_not(img)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    return rotated

def remove_noise_and_smooth(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def extract_text(output_filepath,img):

    if (not output_filepath):
        pass
    elif (output_combobox1.get() == output_combobox1["values"][1]):
        pdf = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')
        with open(output_filepath + ".pdf", 'w+b') as f:
            f.write(pdf)
        messagebox.showinfo("Save Successfull", "Text saved as a PDF file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][2]):
        doc = pytesseract.image_to_string(img)
        with open(output_filepath + ".doc", 'w') as f:
            f.write(doc)
        messagebox.showinfo("Save Successfull", "Text saved as a .doc file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][3]):
        txt = pytesseract.image_to_string(img)
        with open(output_filepath + ".txt", 'w') as f:
            f.write(txt)
        messagebox.showinfo("Save Successfull", "Text saved as a .txt file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][4]):
        xml = pytesseract.image_to_alto_xml(img)
        with open(output_filepath + ".xml", 'w+b') as f:
            f.write(xml)
        messagebox.showinfo("Save Successfull", "Text saved as a XML file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][5]):
        hocr = pytesseract.image_to_pdf_or_hocr(img, extension='hocr')
        with open(output_filepath + ".html", 'w+b') as f:
            f.write(hocr)
        messagebox.showinfo("Save Successfull", "Text saved as a .html file successfully.")

    else:
        messagebox.showerror("Unsupported document type", "This document type is unsupported!")

def save_text_from_credit_card_and_plate(output_filepath,text):

    if (not output_filepath):
        pass
    elif (output_combobox1.get() == output_combobox1["values"][1]):
        with open(output_filepath + ".pdf", 'w+b') as f:
            f.write(text)
        messagebox.showinfo("Save Successfull", "Text saved as a PDF file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][2]):
        with open(output_filepath + ".doc", 'w') as f:
            f.write(text)
        messagebox.showinfo("Save Successfull", "Text saved as a .doc file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][3]):
        with open(output_filepath + ".txt", 'w') as f:
            f.write(text)
        messagebox.showinfo("Save Successfull", "Text saved as a .txt file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][4]):
        with open(output_filepath + ".xml", 'w+b') as f:
            f.write(text)
        messagebox.showinfo("Save Successfull", "Text saved as a XML file successfully.")

    elif (output_combobox1.get() == output_combobox1["values"][5]):
        with open(output_filepath + ".html", 'w+b') as f:
            f.write(text)
        messagebox.showinfo("Save Successfull", "Text saved as a .html file successfully.")

    else:
        messagebox.showerror("Unsupported document type", "This document type is unsupported!")



def start_process():

    """
    try:
        text = pytesseract.image_to_string(img)
    except:
        messagebox.showerror("Error", "Image type is unsupported or there is no text in the image!")
        return

    if (not text.strip()):
        messagebox.showinfo("No text", "Any text didn't found in the image!")
        return 
    """
    if(operation_combobox2.get() == operation_combobox2["values"][1] and not search_entry.get().strip()):
        messagebox.showwarning("Search string", "Search string can not be empty!!")
        return

    if(operation_combobox2.get() == operation_combobox2["values"][2] and 
                output_combobox1.get() == output_combobox1["values"][0]):
        messagebox.showwarning("Output type select", "Please select an output type!!")
        return
        
    global output
    global is_new_image
    global temp_filename
    if (operation_combobox.get() == operation_combobox["values"][1]):

        if(is_new_image):

            temp_filename = set_image_dpi_resize()
            img = cv2.imread(temp_filename)
            img = remove_shadows(img)
            img = remove_noise_and_smooth(img)
            cv2.imshow("removed noise image", img)
            cv2.waitKey(0)
            img = correct_skewness(img)
            cv2.imshow("final image",img)
            cv2.waitKey(0)
            os.remove(temp_filename)
            cv2.imwrite(temp_filename,img)
            is_new_image = False
        else:
            img = cv2.imread(temp_filename)

    elif(operation_combobox.get() == operation_combobox["values"][2]):
        output = []
        if(is_new_image):
            temp_filename = set_image_dpi()
            img = cv2.imread(temp_filename)
            img = imutils.resize(img, width=300)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

            # compute the Scharr gradient of the tophat image, then scale
            # the rest back into the range [0, 255]
            gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                ksize=-1)
            gradX = np.absolute(gradX)
            (minVal, maxVal) = (np.min(gradX), np.max(gradX))
            gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
            gradX = gradX.astype("uint8")
            gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
            thresh = cv2.threshold(gradX, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # apply a second closing operation to the binary image, again
            # to help close gaps between credit card number regions
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
            
            # find contours in the thresholded image, then initialize the
            # list of digit locations
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            locs = []

            for (i, c) in enumerate(cnts):
                # compute the bounding box of the contour, then use the
                # bounding box coordinates to derive the aspect ratio
                (x, y, w, h) = cv2.boundingRect(c)
                ar = w / float(h)
                # since credit cards used a fixed size fonts with 4 groups
                # of 4 digits, we can prune potential contours based on the
                # aspect ratio
                if ar > 2.5 and ar < 4.0:
                    # contours can further be pruned on minimum/maximum width
                    # and height
                    if (w > 40 and w < 55) and (h > 10 and h < 20):
                        # append the bounding box region of the digits group
                        # to our locations list
                        locs.append((x, y, w, h))
            locs = sorted(locs, key=lambda x:x[0])
            
            
            # loop over the 4 groupings of 4 digits
            for (i, (gX, gY, gW, gH)) in enumerate(locs):
                # initialize the list of group digits
                groupOutput = []
                # extract the group ROI of 4 digits from the grayscale image,
                # then apply thresholding to segment the digits from the
                # background of the credit card
                group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
                group = cv2.threshold(group, 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                # detect the contours of each individual digit in the group,
                # then sort the digit contours from left to right
                digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                digitCnts = imutils.grab_contours(digitCnts)
                digitCnts = contours.sort_contours(digitCnts,
                    method="left-to-right")[0]

                # loop over the digit contours
                for c in digitCnts:
                    # compute the bounding box of the individual digit, extract
                    # the digit, and resize it to have the same fixed size as
                    # the reference OCR-A images
                    (x, y, w, h) = cv2.boundingRect(c)
                    roi = group[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (57, 88))
                    # initialize a list of template matching scores	
                    scores = []
                    # loop over the reference digit name and digit ROI
                    for (digit, digitROI) in digits.items():
                        # apply correlation-based template matching, take the
                        # score, and update the scores list
                        result = cv2.matchTemplate(roi, digitROI,
                            cv2.TM_CCOEFF)
                        (_, score, _, _) = cv2.minMaxLoc(result)
                        scores.append(score)
                    # the classification for the digit ROI will be the reference
                    # digit name with the *largest* template matching score
                    groupOutput.append(str(np.argmax(scores)))

                # draw the digit classifications around the group
                cv2.rectangle(img, (gX - 5, gY - 5),
                    (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
                cv2.putText(img, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # update the output digits list
                output.extend(groupOutput)
            
            os.remove(temp_filename)
            if(len(output)==0):
                messagebox.showwarning("Didn't found", "Didn't found any number!")
                return
            cv2.imwrite(temp_filename,img)
            is_new_image = False
            
            print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
            print("Credit Card #: {}".format("".join(output)))
            cv2.imshow("Image", img)
            cv2.waitKey(0)
        else:
            img = cv2.imread(temp_filename)

        

    elif(operation_combobox.get() == operation_combobox["values"][3]):
        if(is_new_image):
            temp_filename = set_image_dpi()
            img = cv2.imread(temp_filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            gray = cv2.bilateralFilter(gray, 13, 15, 15) 

            edged = cv2.Canny(gray, 30, 200) 
            cntrs = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = imutils.grab_contours(cntrs)
            cntrs = sorted(cntrs, key = cv2.contourArea, reverse = True)[:10]
            screenCnt = None

            for c in cntrs:
                
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
                if len(approx) == 4:
                    screenCnt = approx
                    break

            if screenCnt is None:
                detected = 0
                print ("No contour detected")
            else:
                detected = 1

            if detected == 1:
                cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

            mask = np.zeros(gray.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(img,img,mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx+1, topy:bottomy+1]

            output = pytesseract.image_to_string(Cropped, config='--psm 11')
            print("Detected license plate Number is:",output)
            img = cv2.resize(img,(500,300))
            Cropped = cv2.resize(Cropped,(400,200))
            cv2.imshow('car',img)
            cv2.imshow('Cropped',Cropped)
            cv2.waitKey(0) 
            os.remove(temp_filename)
            cv2.imwrite(temp_filename,Cropped)
            is_new_image = False
        else:
            img = cv2.imread(temp_filename)

    if(operation_combobox2.get() == operation_combobox2["values"][1]):

        search_text = search_entry.get().strip().lower()
        search_text_in_document(search_text,img)

    elif(operation_combobox2.get() == operation_combobox2["values"][2]):
        if(operation_combobox.get()==operation_combobox["values"][1]):
            output_filepath = filedialog.asksaveasfilename(initialfile = filepath.split("/")[-1].split(".")[0])
            extract_text(output_filepath,img)
        elif(operation_combobox.get()==operation_combobox["values"][2]):
            output_filepath = filedialog.asksaveasfilename(initialfile = filepath.split("/")[-1].split(".")[0])
            output = "Credit Card Type: {}".format(FIRST_NUMBER[output[0]])+"\n"+"Credit Card #: {}".format("".join(output))
            save_text_from_credit_card_and_plate(output_filepath,output)
        elif(operation_combobox.get()==operation_combobox["values"][3]):
            output_filepath = filedialog.asksaveasfilename(initialfile = filepath.split("/")[-1].split(".")[0])
            save_text_from_credit_card_and_plate(output_filepath,output)
        

    


search_entry = Entry(window,width = 30)
search_entry.insert(0,"Please enter the search text")
start_button = Button(window, text="Start Process",bg='green',command = start_process)

def opcombobox_function(*args):

    search_entry.place_forget()
    output_combobox1.place_forget()
    operation_combobox2.place_forget()
    if (operation_combobox.get() == operation_combobox["values"][0]):
        start_button.place_forget()
        return

    operation_combobox2.place(x=285, y=500)

def opcombobox2_function(*args):
    start_button.place(x=350, y=550)
    search_entry.place_forget()
    output_combobox1.place_forget()


    if (operation_combobox2.get() == operation_combobox2["values"][0]):
        start_button.place_forget()

    elif(operation_combobox2.get() == operation_combobox2["values"][1] ):
        search_entry.place(x=500, y=500)
    else:
        output_combobox1.place(x = 490,y = 500)



variable1 = StringVar(window)

operation_combobox = ttk.Combobox(window, textvariable=variable1)
operation_combobox.config(values=("Please select the image type", "Text-only image", 
                                  "Credit card image","Vehicle plate image"),width = 35)

operation_combobox.current(0)
variable1.trace('w', opcombobox_function)





variable2 = StringVar(window)

operation_combobox2 = ttk.Combobox(window, values = ("Please select an operation","Search for a text on the image",
                                                  "Extract all text from image"),width = 30 , textvariable=variable2)
operation_combobox2.current(0)
variable2.trace('w', opcombobox2_function)

output_combobox1 = ttk.Combobox(window, values = ("Please select an output type for text","searchable PDF (.pdf)",
                                                  "MS Word document (.doc)",
                                                  "Text File (.txt)",
                                                  "XML file (.xml)",
                                                  "HOCR file (.html)"),width = 30)
output_combobox1.current(0)






window.mainloop()