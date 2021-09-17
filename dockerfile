FROM python:3.8-slim-buster
WORKDIR /app
RUN apt-get update && apt-get install gcc -y && apt-get clean
RUN pip install torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers sentence-transformers pandas
RUN pip install protobuf
RUN pip install hdbscan umap-learn click tdqm
RUN pip install modAL nltk
RUN pip install pymongo mongomock
CMD ["bash"]