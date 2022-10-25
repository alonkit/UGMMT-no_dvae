import argparse
import random

import torch
import itertools
import matplotlib.pyplot as plt
from property_handler import property_calc, is_valid_molecule

# my files
from data_preprocess import create_dataset, Dataset
from embedder import Embedder, load_checkpoint_embedder
from embedder_train import fit, get_model_train_params, get_dataloader
from embedding_translator import Translator, weights_init_normal, LambdaLR, ReplayBuffer, Statistics, \
    save_checkpoint, save_checkpoint_ppo, load_checkpoint
from property_handler import smiles2fingerprint, rdkit_no_error_print
from validation import general_validation
from common_utils import set_seed, input2output, get_random_list
from ppo.ppo import PPOTrainer
from train import save_checkpoint_ppo, parse_arguments, validation


def input2output(args, input_batch, E, random_seed_list=None, max_out_len=90, recover_seed=True):
    # prepare input
    input_batch = tuple(data.to(E.device) for data in input_batch)



    random_seed_list = args.seed if random_seed_list is None else random_seed_list
    output_batch = []
    mol_lists = [[] for _ in input_batch]

    scores = []
    for seed in random_seed_list:
        # set seed
        set_seed(seed)
        # embedder encode (METN)
        input_batch_emb, _ = E.forward_encoder(input_batch)


        # embedder decode (decode test = input is <bos> and multi for next char + embedding) (METN)

        for i, mol in enumerate(E.decoder_test(max_len=max_out_len, embedding=input_batch_emb)):
            if is_valid_molecule(mol,args.property):
                old_mol_score = property_calc(E.tensor2string(input_batch[i]), args.property)
                mol_score = property_calc(mol, args.property)
                mol_lists[i].append((E.string2tensor(mol),mol_score - old_mol_score, input_batch_emb[i].detach()))
                scores.append(mol_score - old_mol_score)

    print(sum(scores)/ len(scores))
    for i in range(len(mol_lists)):
        if len(mol_lists[i]) == 0:
            continue
        random.shuffle(mol_lists[i])
        output_batch.append(mol_lists[i][0])




    if recover_seed is True:
        set_seed(args.seed)

    return output_batch

if __name__ =="__main__":
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    rdkit_no_error_print()
    dataset_file_A, dataset_file_B, boundaries = create_dataset(args.property, args.rebuild_dataset)

    E,_ = load_checkpoint(args, device)



    A_train_loader = get_dataloader(E, args, E.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)

    random_seed_list = get_random_list(args.num_retries)

    # epoch = 1
    # fig, ax = plt.subplots()
    # avg_similarity_AB, avg_property_AB, avg_SR_AB, avg_validity_AB, avg_novelty_AB, avg_diversity_AB = \
    #     validation(args, 'mid PPO', E, epoch, boundaries, random_seed_list, fig=fig,
    #                ax=ax)
    #
    # exit()



    # PPO
    E.PPO_prepare()
    ppo_trainer = PPOTrainer(E)
    for epoch in range(20):
        for i, real in enumerate(A_train_loader):
            output_batch = input2output(args, real, E, random_seed_list, max_out_len=100)

            output_batch.sort(key=lambda x: -len(x[0]))
            valid_responses, scores, queries = zip(*output_batch)
            ppo_trainer.step(queries, valid_responses, scores)
            # print("x", end='')
        # run validation
        if args.is_valid is True and (epoch == 1 or epoch % args.validation_freq == 0):
            if epoch == 1:
                fig, ax = plt.subplots()
            avg_similarity_AB, avg_property_AB, avg_SR_AB, avg_validity_AB, avg_novelty_AB, avg_diversity_AB = \
                validation(args, 'mid PPO', E, epoch, boundaries, random_seed_list, fig=fig,
                           ax=ax)
            # save plots
            if args.plot_results is True:
                fig.savefig(args.plots_folder + '/' + args.property + '/mid PPO')

            # early stopping
            current_criterion = avg_SR_AB
            is_early_stop, runs_without_improvement = \
                early_stop(args.early_stopping, current_criterion, best_criterion, runs_without_improvement)
            if is_early_stop:
                break

                # save checkpoint
                best_criterion = save_checkpoint_ppo(current_criterion, best_criterion, None, E, args)

