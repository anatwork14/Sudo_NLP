# dataloader.py
import os
import bs4
import requests
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

class DataLoader:
    def __init__(self, datasource: str):
        self.datasource = datasource

    def load_data(self):
        # Case 1: URL → use WebBaseLoader + BeautifulSoup
        if self.datasource.startswith("http://") or self.datasource.startswith("https://"):
            loader = WebBaseLoader(
                web_paths=(self.datasource,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
                    ),
                ),
            )
            return loader.load()

        # Case 2: Local file → read and wrap in Document
        if not os.path.exists(self.datasource):
            raise ValueError(f"Datasource not found: {self.datasource}")

        with open(self.datasource, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.datasource})]
