from manual_generation.models import Model, SinglePartModel, SimilarityModel
from utils.meters import Meters
from manual_generation.eval import eval_assembly_tree
from tqdm import tqdm
from pprint import pprint
import argparse
from manual_generation.dataset import Dataset


def evaluate(model: Model, dataset: Dataset):
    meters = Meters()
    for f in tqdm(dataset):
        tree_gt = f['tree']
        tree_pred = model(f)
        result = eval_assembly_tree(tree_gt, tree_pred)
        for k in result:
            for k2 in result[k]:
                meters.update(k + '_' + k2, result[k][k2])
    return meters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_features_pkl', type=str)
    parser.add_argument('--data_json', type=str)
    parser.add_argument('--parts_dir', type=str)
    args = parser.parse_args()

    dataset = Dataset(data_json=args.data_json, parts_dir=args.parts_dir, part_features_pkl=args.part_features_pkl)
    evaluate_models = ['single_part', 'similarity']
    # evaluate_models = ['single_part']
    # evaluate_models = ['similarity']
    if 'single_part' in evaluate_models:
        single_part_model = SinglePartModel()
        meters = evaluate(single_part_model, dataset)
        pprint('Single Part Model:')
        pprint(meters.avg_dict())
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')

    if 'similarity' in evaluate_models:
        similarity_model = SimilarityModel()
        meters = evaluate(similarity_model, dataset)
        pprint('Similarity Model:')
        pprint(meters.avg_dict())
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')
