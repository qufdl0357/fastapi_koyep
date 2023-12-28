from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
VECTARA_API_KEY = os.environ.get('VECTARA_API_KEY')
VECTARA_CUSTOMER_ID = os.environ.get('VECTARA_CUSTOMER_ID')
VECTARA_CORPUS_ID = os.environ.get('VECTARA_CORPUS_ID')

def _Insert_Vectara(keyword):
    from apify_client import ApifyClient
    from datetime import datetime
    from langchain.utilities import ApifyWrapper
    from langchain_core.documents.base import Document

    APIFY_API_TOKEN = os.environ.get('APIFY_API_TOKEN')
    client = ApifyClient(APIFY_API_TOKEN)
    run_input = {
        "queries" : keyword,
        "maxPagesPerQuery" : 1,
        "resultPerPage" : 100,
        "customDataFunction" : """
        async({input, $, request, response, html}) => {
            return{
                pageTitle: $('title').text(),
            };
        };
    """
    }
    run = client.actor("apify/google-search-scraper").call(run_input = run_input)
    loader = client.dataset(run["defaultDatasetId"]).iterate_items()
    
    temp = list()
    result = list()

    for i in loader:
        temp.append(i)

    data = temp[0]['organicResults']

    filtered_data = [item for item in data if 'date' in item and datetime.fromisoformat(item['date'][:-1]).year == 2023]

    urls = []
    dates = []

    for item in filtered_data:
        urls.append(item['url'])
        dates.append(item['date'])
    
    min_date = min(dates)

    apify = ApifyWrapper()
    startUrls = [{"url": url} for url in urls]
    run_input = {
        "startUrls": startUrls,
    "includeUrlGlobs": [],
    "excludeUrlGlobs": [],
    "initialCookies": [],
    "proxyConfiguration": { "useApifyProxy": True },
    "removeElementsCssSelector": """nav, footer, script, style, noscript, svg,
    [role=\"alert\"],
    [role=\"banner\"],
    [role=\"dialog\"],
    [role=\"alertdialog\"],
    [role=\"region\"][aria-label*=\"skip\" i],
    [aria-modal=\"true\"]""",
    "clickElementsCssSelector": "[aria-expanded=\"false\"]",
    }

    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input=run_input,
        dataset_mapping_function = lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )

    from langchain.vectorstores import Vectara
    
    vectara = Vectara(
      vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID")
    , vectara_corpus_id   = os.getenv("VECTARA_CORPUS_ID")
    , vectara_api_key     = os.getenv("VECTARA_API_KEY")
    )

    documents = loader.load()

    vectera = Vectara.from_documents(
        documents
        , embedding=None
        , doc_metadata={"category":"knowledgebase"
                    , "keyword":keyword
                    , "date":min_date
                    }
    )


    return "Success Vectara Upload " + keyword


def _Post_Keyword(keyword):
    return keyword