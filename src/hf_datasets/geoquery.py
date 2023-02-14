import json
import datasets


_CITATION = """\
@InProceedings{zelle:aaai96,
title={Learning to Parse Database Queries using Inductive Logic Programming},
author={John M. Zelle and Raymond J. Mooney},
booktitle={AAAI/IAAI},
month={August},
address={Portland, OR},
publisher={AAAI Press/MIT Press},
pages={1050-1055},
url="http://www.cs.utexas.edu/users/ai-lab?zelle:aaai96",
year={1996}
}
@inproceedings{shaw-etal-2021-compositional,
    title = "Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?",
    author = "Shaw, Peter  and
      Chang, Ming-Wei  and
      Pasupat, Panupong  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.75",
    doi = "10.18653/v1/2021.acl-long.75",
    pages = "922--938",
    abstract = "Sequence-to-sequence models excel at handling natural language variation, but have been shown to struggle with out-of-distribution compositional generalization. This has motivated new specialized architectures with stronger compositional biases, but most of these approaches have only been evaluated on synthetically-generated datasets, which are not representative of natural language variation. In this work we ask: can we develop a semantic parsing approach that handles both natural language variation and compositional generalization? To better assess this capability, we propose new train and test splits of non-synthetic datasets. We demonstrate that strong existing approaches do not perform well across a broad set of evaluations. We also propose NQG-T5, a hybrid model that combines a high-precision grammar-based approach with a pre-trained sequence-to-sequence model. It outperforms existing approaches across several compositional generalization challenges on non-synthetic data, while also being competitive with the state-of-the-art on standard evaluations. While still far from solving this problem, our study highlights the importance of diverse evaluations and the open challenge of handling both compositional generalization and natural language variation in semantic parsing.",
}
"""

_DESCRIPTION = """\
The dataset is constructed from
https://github.com/google-research/language/tree/master/language/compgen/nqg
"""

_HOMEPAGE = ""
_LICENSE = ""

_URL = "https://www.dropbox.com/s/zm2gh1o42sucfmp/geoquery.zip?dl=1"


class NL2Bash(datasets.GeneratorBasedBuilder):
    """The NL2Bash dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "funql": datasets.Value("string"),
                    "question": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators"""
        downloaded_files = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files + f"/geoquery/{self.config.name}/train.tsv"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files + f"/geoquery/{self.config.name}/test.tsv"}
            )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath) as fin:
            for i, line in enumerate(fin.readlines()):
                line = line.rstrip()
                cols = line.split("\t")
                entry = {
                    "question": cols[0],
                    "funql": cols[1].replace(' ', ''),
                }
                yield i, entry