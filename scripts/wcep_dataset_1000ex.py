import json
import datasets

_CITATION = """
@inproceedings{gholipour-ghalandari-etal-2020-large,
    title = "A Large-Scale Multi-Document Summarization Dataset from the {W}ikipedia Current Events Portal",
    author = "Gholipour Ghalandari, Demian  and
      Hokamp, Chris  and
      Pham, Nghia The  and
      Glover, John  and
      Ifrim, Georgiana",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.120",
    doi = "10.18653/v1/2020.acl-main.120",
    pages = "1302--1308",
    abstract = "Multi-document summarization (MDS) aims to compress the content in large document collections into short summaries and has important applications in story clustering for newsfeeds, presentation of search results, and timeline generation. However, there is a lack of datasets that realistically address such use cases at a scale large enough for training supervised models for this task. This work presents a new dataset for MDS that is large both in the total number of document clusters and in the size of individual clusters. We build this dataset by leveraging the Wikipedia Current Events Portal (WCEP), which provides concise and neutral human-written summaries of news events, with links to external source articles. We also automatically extend these source articles by looking for related articles in the Common Crawl archive. We provide a quantitative analysis of the dataset and empirical results for several state-of-the-art MDS techniques.",
}
"""

_DESCRIPTION = """
Wikipedia Current Events Portal (WCEP), provides concise and neutral human-written summaries
of news events, with links to external source articles.
"""

_DOCUMENT = "src_str"
_SUMMARY = "tgt_str"


class WCEP(datasets.GeneratorBasedBuilder):
    """WCEP Dataset. - taken from BillSum template """

    # 2.0.0 data source updated to filter near duplicates.
    # 3.0.0  none of the test examples are 'near duplicates' of an example in the
    #   train set AND they dont have the same title, regardless of similarity.
    VERSION = datasets.Version("3.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _DOCUMENT: datasets.Value("string"),
                    _SUMMARY: datasets.Value("string"),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_path = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": "train-1000.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": "test.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": "val.jsonl"},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                d = json.loads(line)
                summary = d['summary'].replace('\n', ' ').strip()
                articles = []
                for art in d['articles']:
                    articles.append(art['text'].replace('\n', ' ').strip())
                articles = list(set(articles))  # to remove duplicates in input - some repeat input
                input_article = ' ||||| '.join(articles)
                yield id_, {
                    "src_str": input_article,
                    "tgt_str": summary
                }
