import os
import torch
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod
from model.cut import networks
from model.cut import utils as util
from .patchnce import PatchNCELoss


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )  # get device name: CPU or GPU
        self.save_dir = os.path.join(
            opt.checkpoints_dir, opt.name
        )  # save all the checkpoints to save_dir
        if (
            opt.preprocess != "scale_width"
        ):  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals

            return grad_hook

        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                setattr(
                    self, "net" + name, torch.nn.DataParallel(net, self.opt.gpu_ids)
                )

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(
                        self.opt.checkpoints_dir, self.opt.pretrained_name
                    )
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from %s" % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}


class CUTModel(BaseModel):
    """This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for CUT model"""
        parser.add_argument(
            "--CUT_mode",
            type=str,
            default="CUT",
            choices="(CUT, cut, FastCUT, fastcut)",
        )

        parser.add_argument(
            "--lambda_GAN",
            type=float,
            default=1.0,
            help="weight for GAN loss：GAN(G(X))",
        )
        parser.add_argument(
            "--lambda_NCE",
            type=float,
            default=1.0,
            help="weight for NCE loss: NCE(G(X), X)",
        )
        parser.add_argument(
            "--nce_idt",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="use NCE loss for identity mapping: NCE(G(Y), Y))",
        )
        parser.add_argument(
            "--nce_layers",
            type=str,
            default="0,4,8,12,16",
            help="compute NCE loss on which layers",
        )
        parser.add_argument(
            "--nce_includes_all_negatives_from_minibatch",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.",
        )
        parser.add_argument(
            "--netF",
            type=str,
            default="mlp_sample",
            choices=["sample", "reshape", "mlp_sample"],
            help="how to downsample the feature map",
        )
        parser.add_argument("--netF_nc", type=int, default=256)
        parser.add_argument(
            "--nce_T", type=float, default=0.07, help="temperature for NCE loss"
        )
        parser.add_argument(
            "--num_patches", type=int, default=256, help="number of patches per layer"
        )
        parser.add_argument(
            "--flip_equivariance",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT",
        )

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False,
                lambda_NCE=10.0,
                flip_equivariance=True,
                n_epochs=150,
                n_epochs_decay=50,
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "D_real", "D_fake", "G", "NCE"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(",")]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ["NCE_Y"]
            self.visual_names += ["idt_B"]

        if self.isTrain:
            self.model_names = ["G", "F", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
        )
        self.netF = networks.define_F(
            opt.input_nc,
            opt.netF,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            self.gpu_ids,
            opt,
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                self.gpu_ids,
                opt,
            )

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(),
                    lr=self.opt.lr,
                    betas=(self.opt.beta1, self.opt.beta2),
                )
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if self.opt.nce_idt and self.opt.isTrain
            else self.real_A
        )
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (
                np.random.random() < 0.5
            )
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[: self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0) :]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = (
                self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            )
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers
        ):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


class SinCUTModel(CUTModel):
    """This class implements the single image translation model (Fig 9) of
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--lambda_R1",
            type=float,
            default=1.0,
            help="weight for the R1 gradient penalty",
        )
        parser.add_argument(
            "--lambda_identity",
            type=float,
            default=1.0,
            help='the "identity preservation loss"',
        )

        parser.set_defaults(
            nce_includes_all_negatives_from_minibatch=True,
            dataset_mode="singleimage",
            netG="stylegan2",
            stylegan2_G_num_downsampling=1,
            netD="stylegan2",
            gan_mode="nonsaturating",
            num_patches=1,
            nce_layers="0,2,4",
            lambda_NCE=4.0,
            ngf=10,
            ndf=8,
            lr=0.002,
            beta1=0.0,
            beta2=0.99,
            load_size=1024,
            crop_size=64,
            preprocess="zoom_and_patch",
        )

        if is_train:
            parser.set_defaults(
                preprocess="zoom_and_patch",
                batch_size=16,
                save_epoch_freq=1,
                save_latest_freq=20000,
                n_epochs=8,
                n_epochs_decay=8,
            )
        else:
            parser.set_defaults(
                preprocess="none",  # load the whole image as it is
                batch_size=1,
                num_test=1,
            )

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if self.isTrain:
            if opt.lambda_R1 > 0.0:
                self.loss_names += ["D_R1"]
            if opt.lambda_identity > 0.0:
                self.loss_names += ["idt"]

    def compute_D_loss(self):
        self.real_B.requires_grad_()
        GAN_loss_D = super().compute_D_loss()
        self.loss_D_R1 = self.R1_loss(self.pred_real, self.real_B)
        self.loss_D = GAN_loss_D + self.loss_D_R1
        return self.loss_D

    def compute_G_loss(self):
        CUT_loss_G = super().compute_G_loss()
        self.loss_idt = (
            torch.nn.functional.l1_loss(self.idt_B, self.real_B)
            * self.opt.lambda_identity
        )
        return CUT_loss_G + self.loss_idt

    def R1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_img,
            create_graph=True,
            retain_graph=True,
        )
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.opt.lambda_R1 * 0.5)
