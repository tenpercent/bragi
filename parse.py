from typing import List, Tuple

from uuid import uuid4

from stog.data.dataset_builder import load_dataset_reader
from stog.data.dataset_readers.amr_parsing.amr import AMR, AMRGraph
from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities
from stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator import FeatureAnnotator
from stog.predictors.stog import STOGPredictor
from stog.models.model import Model
from stog.utils.params import Params
# preprocessing steps
from stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner import clean as clean_amr
from stog.data.dataset_readers.amr_parsing.preprocess.recategorizer import Recategorizer
from stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor import TextAnonymizor
from stog.data.dataset_readers.amr_parsing.preprocess.sense_remover import SenseRemover
# postprocessing steps
from stog.data.dataset_readers.amr_parsing.postprocess.node_restore import NodeRestore
from stog.data.dataset_readers.amr_parsing.postprocess.wikification import Wikification
from stog.data.dataset_readers.amr_parsing.postprocess.expander import Expander


def make_dummy_amr(sentence: str) -> AMR:
    amr = AMR()
    amr.id = uuid4().hex
    amr.sentence = sentence

    amr.graph = AMRGraph.decode('(d / dummy)')
    return amr


def add_annotation(amr: AMR, annotation: dict) -> None:
    amr.tokens = annotation['tokens']
    amr.lemmas = annotation['lemmas']
    amr.pos_tags = annotation['pos_tags']
    amr.ner_tags = annotation['ner_tags']
    amr.graph.set_src_tokens(amr.get_src_tokens())


def setup_pipeline(serialization_dir: str,
                   util_dir: str,
                   stanford_nlp_server_url: str = 'http://localhost:9999',
                   device: str = 'cpu') \
        -> Tuple[FeatureAnnotator, STOGPredictor, \
            Recategorizer, TextAnonymizor, SenseRemover, \
                NodeRestore, Wikification, Expander]:
    params = Params.from_file(f'{serialization_dir}/config.json')
    params.loading_from_archive = False
    weights_file = f'{serialization_dir}/best.th'
    
    model = Model.load(params,
                    weights_file=weights_file,
                    serialization_dir=serialization_dir,
                    device=device)
    predictor = STOGPredictor(model=model, dataset_reader=None)
    annotator = FeatureAnnotator(url=stanford_nlp_server_url,
                                 compound_map_file=f'{util_dir}/joints.txt')
    recategorizer = Recategorizer(util_dir=util_dir)
    text_anonymizor = TextAnonymizor.from_json(f'{util_dir}/text_anonymization_rules.json')
    sense_remover = SenseRemover(node_utils=NodeUtilities.from_json(util_dir, 0))

    node_restorer = NodeRestore(node_utils=NodeUtilities.from_json(util_dir, 0))
    wikification = Wikification(util_dir=util_dir)
    wikification.load_utils()
    expander = Expander(util_dir=util_dir)

    return annotator, predictor, recategorizer, text_anonymizor, sense_remover, \
        node_restorer, wikification, expander


def parse(sentences: List[str],
          serialization_dir: str,
          util_dir: str,
          stanford_nlp_server_url: str = 'http://localhost:9999',
          device: str = 'cpu') -> List[AMR]: 
    
    annotator, predictor, recategorizer, \
        text_anonymizor, sense_remover, \
            node_restorer, wikification, expander = \
                setup_pipeline(serialization_dir, util_dir, stanford_nlp_server_url, device)

    dataset_reader = load_dataset_reader('AMR', word_splitter='bert-base-cased')
    dataset_reader.set_evaluation()  

    predictor._model.set_decoder_token_indexers(dataset_reader._token_indexers)                    
    
    instances = []
    for s in sentences:
        amr = make_dummy_amr(s)
        add_annotation(amr, annotator(s))
        clean_amr(amr)
        recategorizer.recategorize_graph(amr)
        amr.abstract_map = text_anonymizor(amr)
        sense_remover.remove_graph(amr)

        instances.append(dataset_reader.text_to_instance(amr))

    prediction = predictor.predict_batch_instance(instances)

    result = []
    for p in prediction:
        prediction_amr_str = predictor.dump_line(p)
        prediction_amr = list(AMRIO.read_str(prediction_amr_str))[0]
        node_restorer.restore_instance(prediction_amr)
        wikification.wikify_graph(prediction_amr)
        expander.expand_graph(prediction_amr)
        result.append(prediction_amr)

    return result


if __name__ == "__main__":
    raise NotImplementedError
