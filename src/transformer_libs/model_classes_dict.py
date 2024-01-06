from src.transformer_libs import decoder
from src.transformer_libs import new_PT
from src.transformer_libs import LCT
from src.transformer_libs import BNT
from src.transformer_libs import trie_transformer
model_class_dict = {'decoder':decoder.Decoder, 'default_decoder': decoder.Decoder, 'base_decoder': decoder.Decoder,
                    'probabilistic_transformer': new_PT.ProbabilisticTransformer, 'LCT': LCT.LongContextTransformer,
                    'BNT':BNT.BottleneckTransformer, 'trie_transformer': trie_transformer.TrieTransformer}