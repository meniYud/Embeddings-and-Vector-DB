import ingestion
import retrieval
import lcel_retrieval
import ui


def main(ingest: bool = False, retrieve: bool = False):
    if ingest:
        ingestion.main()

    if retrieve:
        retrieval.main()
        # lcel_retrieval.main()

if __name__ == "__main__":
    print("Hello from embeddings-and-vector-db!")
    # main(ingest=False, retrieve=True)
    ui.main()

