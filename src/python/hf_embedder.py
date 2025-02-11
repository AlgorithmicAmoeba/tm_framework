import json
import requests
import timeit
from tqdm import tqdm

class HFEmbedder:
    """HuggingFaceHub embedding models.
    """
    def __init__(self, url: str="http://localhost:3080/embed"):

        self.client = requests.Session()
        self.client.headers.update({"Content-Type": "application/json"})

        self.model_kwargs = None
        self.url = url

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
        responses = self.client.post(
            url=self.url,
            json={"inputs": texts, **_model_kwargs},
        )
        if responses.status_code != 200:
            raise ValueError(f"Failed to get embeddings from HuggingFaceHub: {responses.text}")
        return json.loads(responses.text)
    


def main():
    # Load the HuggingFaceEndpointEmbeddings class
    embedder = HFEmbedder(url="http://localhost:3080")
    # Embed some documents
    # Define the embedding operation
    def embed_test():
        return embedder.embed(
            texts=[
                "This is a test document. " * 10,
                "This is another test document. " * 10,
            ] * 10
        )
    
    # Run the benchmark
    number = 100
    # Show progress during benchmark
    total_time = 0
    for _ in tqdm(range(number), desc="Benchmarking embeddings"):
        start_time = timeit.default_timer()
        embed_test()
        total_time += timeit.default_timer() - start_time
    time_taken = total_time
    
    print(f"Average time per embedding (over {number} runs): {time_taken/number:.4f} seconds")


if __name__ == "__main__":
    main()