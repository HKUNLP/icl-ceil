import json
import datasets


_CITATION = """\
@inproceedings{yin21naacl,
    title = {Compositional Generalization for Neural Semantic Parsing via Span-level Supervised Attention},
    author = {Pengcheng Yin and Hao Fang and Graham Neubig and Adam Pauls and Emmanouil Antonios Platanios and Yu Su and Sam Thomson and Jacob Andreas},
    booktitle = {Meeting of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    address = {Mexico City},
    month = {June},
    url = {https://www.aclweb.org/anthology/2021.naacl-main.225/},
    year = {2021}
}
"""

_DESCRIPTION = """\
The dataset is constructed from
https://github.com/microsoft/compositional-generalization-span-level-attention
"""

_HOMEPAGE = ""
_LICENSE = ""

_URL = "https://www.dropbox.com/s/adrr7bqosqqf5ee/smcalflow_cs.zip?dl=1"


class SMCalFlow_CS(datasets.GeneratorBasedBuilder):
    """The SMCalFlow_CS dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "lispress": datasets.Value("string"),
                    "user_utterance": datasets.Value("string")
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
                name="train",
                gen_kwargs={"filepath": downloaded_files + "/smcalflow_cs/train.json"}
            ),
            datasets.SplitGenerator(
                name="8_shot",
                gen_kwargs={"filepath": downloaded_files + "/smcalflow_cs/8-shot.json"}
            ),
            datasets.SplitGenerator(
                name="16_shot",
                gen_kwargs={"filepath": downloaded_files + "/smcalflow_cs/16-shot.json"}
            ),
            datasets.SplitGenerator(
                name="32_shot",
                gen_kwargs={"filepath": downloaded_files + "/smcalflow_cs/32-shot.json"}
            ),
            datasets.SplitGenerator(
                name="valid_cross",
                gen_kwargs={"filepath": downloaded_files + "/smcalflow_cs/valid-cross.json"}
            ),
            datasets.SplitGenerator(
                name="valid_iid",
                gen_kwargs={"filepath": downloaded_files + "/smcalflow_cs/valid-iid.json"}
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath) as fin:
            data = json.load(fin)
            for i, line in enumerate(data):
                entry = {
                    "lispress": line["lispress"],
                    "user_utterance": line["user_utterance"],
                }
                yield i, entry