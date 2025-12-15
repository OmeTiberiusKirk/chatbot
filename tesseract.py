import pytesseract
from pdf2image import convert_from_path
import re

# --- WINDOWS USERS ONLY ---
# If you didn't add Tesseract/Poppler to your PATH, uncomment and update these lines:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# poppler_path = r'C:\Program Files\poppler-xx\bin'
# --------------------------
poppler_path = "C:/Users/stron/Downloads/poppler-25.12.0/Library/bin"


def read_pdf_with_ocr(pdf_path):
    try:
        print("Converting PDF to images...")
        # Convert PDF to a list of images (one image per page)
        # Windows users: add argument -> poppler_path=poppler_path
        images = convert_from_path(pdf_path, poppler_path=poppler_path)

        full_text = ""

        print(f"Processing {len(images)} pages with OCR...")

        for i, image in enumerate(images):
            # Extract text from the image
            text = pytesseract.image_to_string(image, lang='tha+eng')

            # Formatting distinct pages
            full_text += f"\n--- Page {i + 1} ---\n"
            full_text += text

        return full_text

    except Exception as e:
        return f"Error: {e}"


THAI_MARKS = "่้๊๋ิีึืุูั็์ํเาะโไแใ์"
print(list(THAI_MARKS))


def clean_thai_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"([ก-ฮ])\s+([ก-ฮ])", r"\1\2", text)
    text = re.sub(rf"([ก-ฮ])\s+([{THAI_MARKS}])", r"\1\2", text)
    text = re.sub(rf"([{THAI_MARKS}])\s+([ก-ฮ])", r"\1\2", text)
    text = re.sub(
        rf"([{THAI_MARKS}])\s+([{THAI_MARKS}])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Usage
# pdf_file = "tor1.pdf"
# extracted_text = read_pdf_with_ocr(pdf_file)


# Print or save the text
extracted_text = clean_thai_text("--- Page 1 --- ขอบ เขต ของ ง า น (Terms of Reference : TOR) จ้ า งบ ริก า รบํ า รุงรักษา แล ะ ซ่อม แซม แก้ ไขร ะ บ      บบ ง า นอิ เล็กท รอ นิกส์ของ สํ า นักง า นป ลัดก ร ะ ทร วง คม น า คม ๑ . คว า ม เป็นม า สํ า นักง า นป ลัดก ร ะ ทร            วง คม น า คม ได้ดํ า เนินก า ร โคร งก า รศึกษา ออ ก แบบ ร ะ บบ บูรณา ก า รง า นอิ เล็กท รอ นิกส์ของ กร ะ ทร วง คม          น า คม แล ะ พัฒน า ร ะ บบ ง า นอิ เล็กท รอ นิกส์ของ สํ า นักง า นป ลัดก ร ะ ทร วง คม น า คม กรุง เทพ มห า นค ร ๑ ร        ร ะ บบ ต า มสัญญ า เลข ที่สป ค . ๕ ๓ ๕ / ๒ ๕ ๕ ๕ ลง วันที่ ๓ ๑ มีน า คม wees แล ะ สัญญ า แก้ ไข เพิ่ม เติมฉบับที่ ๑              ๑ สัญญ า เลข ที่สป ค . ๕ ๐ / ๒ ๕ ๒ ๐ ๐ ลง วันที่ ๑ ๓ มิถุน า ยน ๒ ๕ ๒ ๐ เพื่อศึกษา วิ เคร า ะ ห์ออ ก แบบ แล ะ พัฒน               า ร ะ บบ ก า ร ให้บริก า รป ร ะ ชา ชนของ กร ะ ทร วง คม น า คม โดย ก า รบูรณา ก า รก ร ะ บว นง า น (Business Proces  ss) ให้ส า ม า รถ เชื่อม โยง แลก เปลี่ยน ข้อมูลร ะ หว่ า งกัน ได้อย่ า งค รอ บค ลุม แล ะ ใช้ปร ะ โยชน์ร่วม กัน ในก                 า รล ดค ว า มซ้้ า ซ้อน แล ะ เพิ่มค ว า มร วด เร็ว ในก า รป ฏิบัติง า น แล ะ ก า ร ให้บริก า รซึ่งภา ย ใต้ โคร งก               า ร ได้มีก า รจัดห า เครื่อง คอ มพิว เตอ ร์ แม่ข่ า ยจํ า นว น ๒ เครื่อง แล ะ พัฒน า ร ะ บบ ง า นอิ เล็กท รอ นิกส์                จํ า นว น ๒ ๑ ร ะ บบ โดย สํ า นักง า นป ลัดก ร ะ ทร วง คม น า คม ได้ ใช้ ในก า รป ฏิบัติง า นม า อย่ า งต่อ เนื่อง            ง แล ะ ร ะ บบ บริห า รง า นส า รบ รร ณอิ เล็กท รอ นิกส์ ในก า รนี้ เพื่อ ให้ เครื่อง คอ มพิว เตอ ร์ แม่ข่ า ย แล ะ")

# Optional: Save to a text file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)
