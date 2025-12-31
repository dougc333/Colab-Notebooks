from nemo.collections import llm

if __name__ == '__main__':
    llm.import_ckpt(
        model=llm.LlamaEmbeddingModel(config=llm.Llama32EmbeddingConfig1B()),
        source="hf://meta-llama/Llama-3.2-1B",
    )


