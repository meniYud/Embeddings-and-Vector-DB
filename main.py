
import ingestion
import retrieval



def main(ingest: bool = False, retrieve: bool = False):
    if ingest:
        ingestion.main()

    if retrieve:
        retrieval.main()

if __name__ == "__main__":
    print("Hello from embeddings-and-vector-db!")
    main(ingest=False, retrieve=True)

