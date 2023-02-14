import json
import datasets


_CITATION = """\
@inproceedings{LinWZE2018:NL2Bash, 
  author = {Xi Victoria Lin and Chenglong Wang and Luke Zettlemoyer and Michael D. Ernst}, 
  title = {NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System}, 
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources
               and Evaluation {LREC} 2018, Miyazaki (Japan), 7-12 May, 2018.},
  year = {2018} 
}
"""

_DESCRIPTION = """\
The dataset is constructed from
https://github.com/TellinaTool/nl2bash
"""

_HOMEPAGE = ""
_LICENSE = ""

_URL = "https://www.dropbox.com/s/wy7uahzbir7lrq1/nl2bash.zip?dl=1"


class NL2Bash(datasets.GeneratorBasedBuilder):
    """The NL2Bash dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "nl": datasets.Value("string"),
                    "bash": datasets.Value("string")
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
                gen_kwargs={"filepath": downloaded_files + "/nl2bash/train.json"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files + "/nl2bash/dev.json"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files + "/nl2bash/test.json"}
            )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath) as fin:
            data = json.load(fin)
            for i, line in enumerate(data):
                entry = {
                    "nl": line["nl"],
                    "bash": line["bash"],
                }
                yield i, entry