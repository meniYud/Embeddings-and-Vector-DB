   
import chatty_pdf_ingestion
import chatty_pdf_retieval

def main(ingestion: bool = False, retrieval: bool = False):
    if ingestion:
        chatty_pdf_ingestion.main()
    if retrieval:
        chatty_pdf_retieval.main()

if __name__ == "__main__":
    main(ingestion=True, retrieval=True)

