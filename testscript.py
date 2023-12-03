
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from tools.frontend_helper import get_model, initialize_db_client

_ = initialize_db_client()
model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=384)
query = "leave policy"
query_encode = [list(i) for i in model.encode([query])]
print(utility.list_collections())
collection = Collection('dhipaya_main_doc')
collection.load()
documents = collection.search(data=query_encode, anns_field="embeddings", param={"metric":"IP","offset":0},
                  output_fields=["text_to_encode", "Term_th", "Meaning_th"], limit=4)
print("no. of retrieved docs", len(documents[0]))

i = 1
for doc in documents[0]:
    # print(doc[0].text)
    print(f'doc {i}')
    print(f'text: \n', doc, "\n")
    # print(f'metadata: \n', doc.metadata, "\n")
    i += 1
