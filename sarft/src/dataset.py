"""PKU-Alianment Dataset."""

import json
import os
import random
import datasets
from hashlib import md5
import ast

logger = datasets.logging.get_logger(__name__)
ANSWER_PREFIX = 'assistant'
SINGLE_QUOTES_SUBSTITUTE = "#$%#"

def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               data_args.instruction_file + data_args.instruction_strategy + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class DataConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """
    
    def __init__(
            self,
            *args,
            data_dir=None,
            over_sampling=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.over_sampling = over_sampling


class DataInstructions(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = DataConfig
    BUILDER_CONFIGS = [
        DataConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "prompt_role": datasets.Value("string"),
                    "prompt_safety": datasets.Value("string"),
                    "is_harmful": datasets.Value("int")
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            logger.error("Please provide right input: data_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir + '/train.json',
                    "subset": "train"
                })
        ]


    def _load_dataset(self, dataset_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)

        return instances

    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances

    def load_dataset(self, dataset_path, subset):

        data = self._load_dataset(dataset_path)
        dataset_name = str(dataset_path) if type(dataset_path) is not str else dataset_path
        role = dataset_name.split('/train.json')[0].split('/')[-1]
        print("role: \n", role)
        prompt_path = "SaRFT/sarft/config/prompt_v2.json"
        with open(prompt_path, encoding="utf-8") as prompt_f:
            prompt_dict = json.load(prompt_f)
        prompt_role = prompt_dict[role]['role']
        prompt_safety = prompt_dict[role]['safe']
        
        for idx, instance in enumerate(data):
            example = {
                "id": str(idx),
                "input": instance['input'],
                "output": instance['output'],
                "subset": subset,
                "prompt_role": prompt_role,
                "prompt_safety": prompt_safety,
                "is_harmful": instance['is_harmful'] if 'is_harmful' in instance else -1
            }
            yield example

    def _generate_examples(self, path=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")

        # load dataset
        idx = -1
        instances = []
        for sample in self.load_dataset(path, subset):
            idx += 1
            instances.append(sample)
            # print("sample: \n", sample)
            yield f"{path}##{idx}", sample
