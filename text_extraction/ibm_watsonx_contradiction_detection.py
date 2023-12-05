from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from tools.frontend_helper import get_model, initialize_db_client
from tools.translation import translate_large_text, translate_to_thai
from tools.frontend_helper import get_db_results_changes


load_dotenv()
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
environment = os.getenv("ENVIRONMENT", "local")

script_dir = os.path.dirname(__file__)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print(
        "Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

model_params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 300,
    # GenParams.RANDOM_SEED: 42,
    # GenParams.TEMPERATURE: 0.7,
    GenParams.REPETITION_PENALTY: 1,
}
model_llm = Model(ModelTypes.LLAMA_2_70B_CHAT.value, params=model_params, credentials=creds, project_id=project_id)


knowledge_based_template = (
    open("assets/bonus/llama2-contradiction-detection.txt",
        encoding="utf8").read().format(
    )
)
custom_prompt = PromptTemplate(template=knowledge_based_template,
                            input_variables=["context", "documents"])

incoming_text = """
คำสั่งที่ ห. 92 /2566
เรื่อง แก้ไขกฎ 
ตามที่เดิมที บริษัทฯ จะเป็นผู้กำหนดวันหยุดประจำปีให้กับพนักงานแต่ละคนตาม หรือ อาจจะกำหนด วันหยุดพักผ่อนประจำปีโดยพิจารณาจากคำขอหยุดพักผ่อนประจำปีของพนักงาน ประกอบกับ ความจำเป็นของงานตามตำแหน่ง โดยการอนุมัติของผู้บังคับบัญชา ตั้งแต่ระดับผู้อำนวยการฝ่ายขึ้นไป บริษัทฯ จะปรับกฏให้การอนุมัติ กระทำโดยอำนาจของผู้บังคับบัญชา ตั้งแต่ระดับผู้จัดการฝ่ายขึ้นไป รวมถึงต้องให้ผู้บังคับบัญชาตั้งแต่ระดับผู้อำนวยการฝ่ายขึ้นไปลงนามอนุมัติด้วย 
จึงเรียนให้ทราบโดยทั่วกัน นอกเหนื่อจากนี้พนักงานที่ทำงานน้อยกว่า 5 ปีจะสามารถหยุดได้เพิ่มอีก 1 วัน จาก 20 เป็น 21 วัน
มีผลบังคับ ตั้งแต่วันที่ 30 พฤศจิกายน 2566 
สั่ง ณ วันที่ 35 ตุลาคม 2566
"""


_ = initialize_db_client()
model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=384)
documents = get_db_results_changes(incoming_text, model, 'dhipaya_changes_policy', 4)


changes_combined = ""
for i in range(len(documents)):
    changes_combined += f"Changes {i} \n" + documents[i].text_to_encode + "\n"

formated_prompt = custom_prompt.format(context=translate_large_text(changes_combined,translate_to_thai, False, max_length=250),
                                        documents=translate_large_text(incoming_text,translate_to_thai, False, max_length=250))

print(formated_prompt)
output = model_llm.generate(formated_prompt)
print(output)