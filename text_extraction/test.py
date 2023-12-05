
import re

text = """
คำสั่งที่ ห. 90 / 2566
เรื่อง แก้ไขกฎ 
ตามที่เดิมที บริษัทฯ จะเป็นผู้กำหนดวันหยุดประจำปีให้กับพนักงานแต่ละคนตาม หรือ อาจจะกำหนด วันหยุดพักผ่อนประจำปีโดยพิจารณาจากคำขอหยุดพักผ่อนประจำปีของพนักงาน ประกอบกับ ความจำเป็นของงานตามตำแหน่ง โดยการอนุมัติของผู้บังคับบัญชา ตั้งแต่ระดับผู้อำนวยการฝ่ายขึ้นไป บริษัทฯ จะปรับกฏให้การอนุมัติ กระทำโดยอำนาจของผู้บังคับบัญชา ตั้งแต่ระดับผู้จัดการฝ่ายขึ้นไป 
จึงเรียนให้ทราบโดยทั่วกัน 
มีผลบังคับ ตั้งแต่วันที่ 6 พฤศจิกายน 2566 
สั่ง ณ วันที่ 25 ตุลาคม 2566
"""

# Function to extract dates using a regular expression
def extract_dates(text):
    pattern = r"ตั้งแต่วันที่ (\d{1,2} \S+ \d{4})|สั่ง ณ วันที่ (\d{1,2} \S+ \d{4})"
    matches = re.findall(pattern, text)
    return [date for match in matches for date in match if date]

# Function to extract commands with 'คำสั่งที่' followed by text until 'เรื่อง'
def extract_commands(text):
    pattern = r"คำสั่งที่ (.*?)เรื่อง"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

# Function to extract subjects with 'เรื่อง' followed by text until a newline
def extract_subjects(text):
    pattern = r"เรื่อง (.*?)\n"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

# Extract and print the dates, commands, and subjects
dates = extract_dates(text)
print("Dates:", dates)

commands = extract_commands(text)
print("Commands:", commands)

subjects = extract_subjects(text)
print("Subjects:", subjects)