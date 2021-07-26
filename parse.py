from typing import List

from stog.data.dataset_builder import load_dataset_reader
from stog.data.dataset_readers.amr_parsing.amr import AMR, AMRGraph
from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities
from stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator import FeatureAnnotator
from stog.data.tokenizers.bert_tokenizer import AMRBertTokenizer
from stog.predictors.stog import STOGPredictor
from stog.models import STOG
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


def make_amr(annotation, sentence: str) -> AMR:
    amr = AMR()
    amr.id = '0'
    amr.sentence = sentence

    amr.tokens = annotation['tokens']
    amr.lemmas = annotation['lemmas']
    amr.pos_tags = annotation['pos_tags']
    amr.ner_tags = annotation['ner_tags']

    amr.graph = AMRGraph.decode('(d / dummy)')
    amr.graph.set_src_tokens(amr.get_src_tokens())
    return amr

def parse(sentences: List[str],
          serialization_dir: str,
          util_dir: str,
          stanford_nlp_server_url: str = 'http://localhost:9999',
          device: str = 'cpu') -> List[AMR]: 
    
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
    dataset_reader = load_dataset_reader('AMR', word_splitter='bert-base-cased')
    dataset_reader.set_evaluation()         
    predictor._model.set_decoder_token_indexers(dataset_reader._token_indexers)                    
    node_restorer = NodeRestore(node_utils=NodeUtilities.from_json(util_dir, 0))
    wikification = Wikification(util_dir=util_dir)
    wikification.load_utils()
    expander = Expander(util_dir=util_dir)

    instances = []
    for s in sentences:
        annotation = annotator(s)
        amr = make_amr(annotation, s)
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
