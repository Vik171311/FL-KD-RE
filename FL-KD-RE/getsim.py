
from utils import *

from partition import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_sim(args,pre_model,pre_model_name,train_loader):
    args.maxC = args.C
    args.maxflop_C = args.flop_C
    results = {}


    for i in range(len(pre_model)):
        model = copy.deepcopy(pre_model[i]).to(device)
        feature_layers = dict()
        feature_layers_names = []
        hook_class = GetFeatureHook
        block_list = MODEL_BLOCKS[pre_model_name[i]]
        for name, module in model.named_modules():
            for blk_name in block_list:
                if name.endswith(blk_name):
                    feature_layers_names.append(name)
                    feature_layers[name] = hook_class(module)
                    break

        for data, label in train_loader:
            data = data.to(device)
            #label = label.to(device)
            output = model(data)
        feat = dict()
        size = dict()
        for layer_name in feature_layers_names:
            layer = feature_layers[layer_name]
            layer.concat()
            num = layer.feature.size(0)
            size[layer_name] = [layer.in_size, layer.out_size]
            feat[layer_name] = layer.feature.view(num, -1)

        # for idx, layer_name in enumerate(feature_layers_names):
        #     layer = feature_layers[layer_name]
        #     layer.concat()
        #     num = layer.feature.size(0)
        #     size[layer_name] = [layer.in_size, layer.out_size]
        #     feat[layer_name] = layer.feature.view(num, -1)
        #     if idx == len(MODEL_BLOCKS[pre_model_name[i]]) - 1:
        #         data = feat[layer_name]
        #         density = cosine_distances(data)
        #         non_diagonal_indices = ~np.eye(density.shape[0], dtype=bool)
        #         average_similarity = np.mean(density[non_diagonal_indices])
        #         print(average_similarity)
        #         if average_similarity < args.yuedayuehaodezhi:
        #             mingzi.append(pre_model_name[i])

        result = dict(model_name=pre_model_name[i], size=size, feat=feat)
        results[i] = result

    results_sims = {}
    PYTHS = [i for i in results.keys()]
    pkls_comb = list(combinations(PYTHS, 2))
    pkls_comb += [(pkl, pkl) for pkl in PYTHS]
    for pickle1, pickle2 in reversed(pkls_comb):
        data1 = results[pickle1]
        data2 = results[pickle2]
        name1 = data1['model_name']
        name2 = data2['model_name']
        arch1 = name1
        arch2 = name2

        print(f'Computing {name1}.{name2} similarity')
        # sim = SIM_FUNC['cka'](data1, data2, bs=2048)
        sim = SIM_FUNC['rbf_cka'](data1, data2, bs=2048)
        # sim = SIM_FUNC['lr'](data1, data2, bs=2048)

        results_sim = dict(sim=sim,
                           model1=dict(arch=arch1, model_name=name1),
                           model2=dict(arch=arch2, model_name=name2))

        results_sims[f'{name1}.{name2}'] = results_sim

        del data1
        del data2
    print(results_sims)
    MODEL_INOUT_SHAPE = {}
    for pickle in PYTHS:
        data = results[pickle]
        name = data['model_name']
        MODEL_INOUT_SHAPE[name] = dict(in_size=dict(), out_size=dict())
        for key in data['size'].keys():
            for layer in MODEL_BLOCKS[name]:
                if key.endswith(layer):
                    in_size, out_size = data['size'][key]
                    MODEL_INOUT_SHAPE[name]['in_size'][layer] = tuple(in_size)
                    MODEL_INOUT_SHAPE[name]['out_size'][layer] = tuple(out_size)

        del data

    block_sims = get_all_sim(results_sims)
    all_sim = 0
    for k in range(args.trial):
        block_split_dict = init_partition(args)

        all_assignmemt = init_assign(args, block_split_dict, block_sims)

        non_improved = 0

        for i in range(args.num_iter):
            block_split_dict, _ = repartition(
                args, block_split_dict, block_sims, all_assignmemt)

            all_assignmemt = recenter(
                args, block_split_dict, block_sims, all_assignmemt)

            all_assignmemt = reassign(
                args, block_split_dict, block_sims, all_assignmemt)

            current_sim = total_cost(all_assignmemt, block_sims)

            if current_sim > all_sim:
                all_sim = current_sim
                best_block_split_dict = block_split_dict
                best_all_assignmemt = all_assignmemt
            else:
                non_improved += 1

            # if no improvement of 20 steps, stop
            # if no improvement of 40 steps, stop
            if non_improved > 40:
                print(f'No improvement for {non_improved} iteration')
                print(
                    f'[Trial {k}]: Best Total Similarity {all_sim}, Current Similarity {current_sim}')
                break
    print_partition(best_block_split_dict)
    best_all_assignmemt.get_size(MODEL_INOUT_SHAPE)
    best_all_assignmemt.print_assignment()

    return best_all_assignmemt,best_block_split_dict