#! pip install llama-index

import logging
import sys
import pandas as pd
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI

client = OpenAI(
    api_key = "",
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

reader = SimpleDirectoryReader("../Example_Texts/")
documents = reader.load_data()

data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()
print(eval_questions)
