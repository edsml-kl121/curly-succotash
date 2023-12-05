from PIL import Image
import pytesseract

img_path = 'assets/img/img2.png'
txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg, 'tha')

print(text)