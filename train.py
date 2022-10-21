import argparse
import torch
import itertools
import matplotlib.pyplot as plt
from property_handler import property_calc

# my files
from data_preprocess import create_dataset, Dataset
from embedder import Embedder
from embedder_train import fit, get_model_train_params, get_dataloader
from embedding_translator import Translator, weights_init_normal, LambdaLR, ReplayBuffer, Statistics, \
    save_checkpoint
from property_handler import smiles2fingerprint, rdkit_no_error_print
from validation import general_validation
from common_utils import set_seed, input2output, get_random_list
from ppo.ppo import PPOTrainer

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Unpaired Generative Molecule-to-Molecule Translator'
    )

    # end-end model settings
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train the end-end model')
    parser.add_argument('--epoch_init', type=int, default=1, help='initial epoch')
    parser.add_argument('--epoch_decay', type=int, default=90,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for end-end model')
    parser.add_argument('--property', type=str, default='QED',
                        help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--init_lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--is_valid', default=True, action='store_true', help='run validation every train epoch')
    parser.add_argument('--valid_direction', type=str, default='AB',
                        help='direction of validation translation- AB: A->B; BA: B->A')
    parser.add_argument('--plot_results', default=True, action='store_true',
                        help='plot validation set results during end-end model training')
    parser.add_argument('--print_results', default=True, action='store_true',
                        help='print validation results during end-end model training')
    parser.add_argument('--rebuild_dataset', default=False, action='store_false', help='rebuild dataset files')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints',
                        help='name of folder for checkpoints saving')
    parser.add_argument('--plots_folder', type=str, default='plots_output', help='name of folder for plots saving')
    parser.add_argument('--early_stopping', type=int, default=15, help='Whether to stop training early if there is no\
                        criterion improvement for this number of validation runs.')
    parser.add_argument('--seed', type=int, default=50, help='base seed')
    parser.add_argument('--num_retries', type=int, default=20, help='number of retries for each validation sample')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_property_val', type=int, default=0.8, help='minimal property value for success')
    parser.add_argument('--validation_max_len', type=int, default=90, help='length of validation smiles')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='batch size for validation end-end model')
    parser.add_argument('--validation_freq', type=int, default=3, help='validate every n-th epoch')
    parser.add_argument('--is_CDN', default=False, action='store_false', help='trains CDN network')
    parser.add_argument('--tokenize', default=False, action='store_false', help='use atom tokenization')
    parser.add_argument('--cycle_loss', default=True, action='store_true', help='use cycle loss during training or not')

    # Ablation
    parser.add_argument('--gan_loss', default=False, action='store_true', help='use gan loss during training or not')
    parser.add_argument('--kl_loss', default=True, action='store_true', help='use kl loss during training or not')
    parser.add_argument('--swap_cycle_fp', default=False, action='store_true',
                        help='swap fp in second translator during training or not')
    parser.add_argument('--use_fp', default=True, action='store_true', help='does translator use molecule fp')
    parser.add_argument('--use_EETN', default=True, action='store_true', help='use EETN network')
    parser.add_argument('--conditional', default=False, action='store_true', help='using only fp for optimization')
    parser.add_argument('--no_pre_train', default=False, action='store_true', help='disable METNs pre training')

    args = parser.parse_args()
    return args


def train_iteration_T(real, E, T, optimizer_T, args):
    optimizer_T.zero_grad()

    # embedder (METN)
    real_emb, kl_loss = E.forward_encoder(real)
    if args.kl_loss is False:
        kl_loss = None

    # prepare fingerprints
    if args.use_fp:
        real_fp_str = [smiles2fingerprint(E.tensor2string(mol), fp_translator=True) for mol in real_A]
        real_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_fp_str]).to(device)
        real_fp = real_fp.detach()
    else:
        real_fp = None

    fake_emb = T(real_emb, real_fp)

    cycle_loss,_ = E.forward_decoder(real, fake_emb)

    # Total loss
    if args.cycle_loss and args.gan_loss is False and args.kl_loss is False:  # Main model: only cycle
        loss = cycle_loss
    elif args.cycle_loss and args.kl_loss:  # for ablation: cycle + gan + kl (for stability)
        loss = 2 * cycle_loss + 0.2 * kl_loss
    else:
        print('No such setting for the main model, nor for the ablation tests')
        exit()

    loss.backward()
    optimizer_T.step()

    return loss, cycle_loss, kl_loss, fake_emb


# for ablation
def train_iteration_D(real, fake_emb, model, D, loss_GAN, optimizer_D):
    optimizer_D.zero_grad()

    real_emb, kl_loss = model.forward_encoder(real)

    # Real loss
    pred_real = D(real_emb.detach())
    loss_D_real = loss_GAN(pred_real, torch.ones(pred_real.size(), device=device, requires_grad=False))

    # Fake loss
    pred_fake = D(fake_emb.detach())
    loss_D_fake = loss_GAN(pred_fake, torch.zeros(pred_fake.size(), device=device, requires_grad=False))

    # Total loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    loss_D.backward()
    optimizer_D.step()

    return loss_D, loss_D_real, loss_D_fake


def validation(args, model_name, E, T, epoch, boundaries, random_seed_list, fig=None, ax=None):
    # evaluation mode
    E.eval()
    T.eval()

    # dataset loader
    valid_loader = get_dataloader(E, args, E.dataset.validset, batch_size=args.validation_batch_size,
                                  shuffle=False)

    # number samples in validset
    validset_len = len(E.dataset.validset)

    # tensor to molecule smiles
    def input_tensor2string(input_tensor):
        return E.tensor2string(input_tensor)

    trainset = set(E.dataset.trainset)

    # generate output molecule from input molecule
    def local_input2output(input_batch):
        return input2output(args, input_batch, E, T, E, random_seed_list,
                            max_out_len=args.validation_max_len)

    # use general validation function
    avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity = \
        general_validation(args, local_input2output, input_tensor2string, boundaries, valid_loader, validset_len,
                           model_name, trainset, epoch,
                           fig=fig, ax=ax)

    # back to train mode
    E.train()
    T.train()

    return avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity


def early_stop(early_stopping, current_criterion, best_criterion, runs_without_improvement):
    if early_stopping is not None:
        # first model or best model so far
        if best_criterion is None or current_criterion > best_criterion:
            runs_without_improvement = 0
        # no improvement
        else:
            runs_without_improvement += 1
        if runs_without_improvement >= early_stopping:
            return True, runs_without_improvement  # True = stop training
        else:
            return False, runs_without_improvement


if __name__ == "__main__":

    # parse arguments
    args = parse_arguments()

    # set seed
    set_seed(args.seed)

    # set device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # disable rdkit error messages
    rdkit_no_error_print()

    # epochs for METN pre training
    if args.property is 'QED':
        embedder_epochs_num = 1
    elif args.property is 'DRD2':
        embedder_epochs_num = 12
    else:
        print("property must bt 'QED 'or 'DRD2'")
        exit()

    if args.conditional or args.no_pre_train:
        embedder_epochs_num = 0

    if args.is_CDN is True:
        _, _, boundaries = create_dataset(args.property, rebuild_dataset=False)
        dataset_CDN = Dataset('dataset/' + args.property + '/CDN/CDN')
        model_CDN = Embedder(dataset_CDN, 'CDN', args).to(device)
        embedder_epochs_num = args.epochs
        fit(args, model_CDN, embedder_epochs_num, boundaries, is_validation=True)
        exit()

    # prepare dataset
    dataset_file_A, dataset_file_B, boundaries = create_dataset(args.property, args.rebuild_dataset)
    dataset_A = Dataset(dataset_file_A, use_atom_tokenizer=args.tokenize, isB=False)

    # create  and pre-train the embedders (METNs)
    embedder_epochs_num = 120

    E = Embedder(dataset_A, 'Embedder A', args).to(device)
    # fit(args, E, embedder_epochs_num, boundaries, is_validation=True)



    # create embedding translators (EETN)
    T = Translator().to(device)

    # weights
    T.apply(weights_init_normal)

    # optimizer
    optimizer_T = torch.optim.Adam(itertools.chain(T.parameters(), get_model_train_params(E)), lr=args.init_lr,
                                   betas=(0.5, 0.999))

    # scheduler
    lr_scheduler_T = torch.optim.lr_scheduler.LambdaLR(optimizer_T, lr_lambda=LambdaLR(args.epochs, args.epoch_init,
                                                                                       args.epoch_decay).step)
    # schedulers for ablation

    # train dataloaders
    A_train_loader = get_dataloader(E, args, E.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)

    # for early stopping
    best_criterion = None
    runs_without_improvement = 0

    # generate random seeds
    random_seed_list = get_random_list(args.num_retries)

    # ###### Training ######
    # for epoch in range(args.epoch_init, args.epochs + 1):
    #     print(' ')
    #     print('epoch #' + str(epoch))
    #
    #     # statistics
    #     stats = Statistics()
    #
    #     for i, real_A in enumerate(A_train_loader):
    #         # update translators (EETN) and embedders (METNs)
    #         loss, cycle_loss, kl_loss, fake_emb = \
    #             train_iteration_T(real_A, E, T, optimizer_T, args)
    #
    #         # update statistics
    #         stats.update(loss, cycle_loss, kl_loss)
    #
    #     # print epoch's statistics
    #     stats.print()
    #
    #     # run validation
    #     if args.is_valid is True and (epoch == 1 or epoch % args.validation_freq == 0):
    #         if epoch == 1:
    #             fig, ax = plt.subplots()
    #         avg_similarity_AB, avg_property_AB, avg_SR_AB, avg_validity_AB, avg_novelty_AB, avg_diversity_AB = \
    #             validation(args, 'PRE PPO', E, T, epoch, boundaries, random_seed_list, fig=fig,
    #                        ax=ax)
    #         # save plots
    #         if args.plot_results is True:
    #             fig.savefig(args.plots_folder + '/' + args.property + '/PRE PPO')
    #
    #         # early stopping
    #         current_criterion = avg_SR_AB
    #         is_early_stop, runs_without_improvement = \
    #             early_stop(args.early_stopping, current_criterion, best_criterion, runs_without_improvement)
    #         if is_early_stop:
    #             break
    #
    #         # save checkpoint
    #         best_criterion = save_checkpoint(current_criterion, best_criterion, T, E, args)
    #
    #     # update learning rate
    #     lr_scheduler_T.step()


    # PPO
    E.PPO_prepare()
    ppo_trainer = PPOTrainer(E)
    for i, real in enumerate(A_train_loader):
        query, _ = E.forward_encoder(real)

        # query = query[0:1] # for debug

        # prepare fingerprints
        # if args.use_fp:
        #     fp_str = [smiles2fingerprint(E.tensor2string(mol), fp_translator=True) for mol in real]
        #     fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in fp_str]).to(device)
        #     fp = fp.detach()
        # else:
        #     fp = None
        #
        # query = T(real_emb, fp)
        response = E.decoder_test(100, query)
        response = [E.string2tensor(s) for s in response]
        # remove invalid and get scores
        queries, valid_responses, scores = [], [], []
        lst = []
        for real, fake in zip(query,response):
            try:
                old_score = property_calc(real, args.property)
                score = property_calc(fake, args.property) - old_score
                lst.append((fake, score, real))
                # valid_responses.append(fake)
                # scores.append(score)
                # queries.append(real)
            except:
                # pass
                # valid_responses.append(fake)
                # scores.append(0)
                # queries.append(real)
                lst.append((fake, 0, real))
        lst.sort(key=lambda x: -len(x[0]))
        valid_responses, scores, queries = zip(*lst)
        ppo_trainer.step(queries, valid_responses, scores)






