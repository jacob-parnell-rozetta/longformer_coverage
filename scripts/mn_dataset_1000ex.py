import json
import datasets

_CITATION = """
@inproceedings{fabbri2019,
    title = "Multi-News: A Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model",
    author = "Fabbri, Alexander  and
      Li, Irene  and
      She, Tianwei  and
      Li, Suyi  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1102",
    doi = "10.18653/v1/P19-1102",
    pages = "1074--1084"
}
"""

_DESCRIPTION = """
ABSTRACT:
Automatic generation of summaries from multiple news articles is a valuable tool as the number of online publications 
grows rapidly. Single document summarization (SDS) systems have benefited from advances in neural encoder-decoder model 
thanks to the availability of large datasets. However, multi-document summarization (MDS) of news articles has been 
limited to datasets of a couple of hundred examples. In this paper, we introduce Multi-News, the first large-scale 
MDS news dataset. Additionally, we propose an end-to-end model which incorporates a traditional extractive 
summarization model with a standard SDS model and achieves competitive results on MDS datasets. We benchmark several 
methods on Multi-News and hope that this work will promote advances in summarization in the multi-document setting."
"""

_DOCUMENT = "src_str"
_SUMMARY = "tgt_str"


class MN(datasets.GeneratorBasedBuilder):
    """MN Dataset. - taken from BillSum template """

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
                document = d['document'].replace('\n', ' ').strip()
                yield id_, {
                    "src_str": document,
                    "tgt_str": summary
                }
