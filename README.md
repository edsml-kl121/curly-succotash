# dhipaya-hr-policy

## Building  Apps with watsonx.ai and Streamlit
So I'm guessing you've been hearing a bit about watsonx. Well...now you can build your very own app with it🙌 (I know...crazy right?!). In this tutorial you'll learn how to build your own LLM powered Streamlit with the Watson Machine Learning library.

## Startup 🚀
1. Open your terminal or console window
2. cd into this lab's base directory
3. Copy your .env file into this lab's base folder
4. Add HuggingFaceToken to .env file. See HuggingFace API Setup section below.
5. Run the app by running the command `streamlit run app.py`.

## HuggingFace API Setup
1. Sign In or Sign Up on https://huggingface.co/.
2. Go on Profile (avatar) and click on "Settings".
3. Go to "Access Token" and click on "New token".
4. Give a user friendly name to the token and permission=write. Then click on on generate token.
5. Copy the token and add it in your .env file. `HUGGINGFACEHUB_API_TOKEN=<your_new_huggingface_access_token>`


## Ingest initial data
please ensure you already set .env with following key MILVUS_HOST and MILVUS_PORT
```shell
python3 backend.py
```
## build image
```shell
podman build -t dhipaya-neuralseek:v1.0.0 .
```

## run container
```shell
podman run -v ./.env:/usr/src/app/.env -d -p 8443:8080 dhipaya-neuralseek:v1.0.0
```
