clear all
close all
clc

%add filepath and sub directories
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));


disp('================================')
disp('start HMM decoding')
disp('================================')
tic

%addpath('/packages/speech_recognition/matlab');
%wave_list = '/data/rvg_new/lists/digits.waves'; % 8 kHz MFCCs
%wave_list = '/data/rvg_new/lists/digits_16k.waves'; % 16 kHz DFT based neural net



%********************************************************
%insert neural net names:
%********************************************************

%name_matfile = 'MLP_5layer_2017-05-10_16:39_0.497.mat';
%name_matfile = 'MLP_5layer_2017-05-10_16:39_0.497.mat';
%name_matfile = 'MLP_5layer_2017-05-20_15:44_0.560.mat';
%name_matfile = 'MLP_5layer_bnormentry_fft_2017-06-09_12:56_0.398.mat';
%name_matfile = 'MLP_5lay_7fs_257coeffs_512nodes_135class_20eps_relu2017-06-19_14:50_0.449.mat';
%name_matfile = 'MLP_5lay_11fs_257coeffs_512nodes_135class_20eps_relu2017-06-20_08:58_0.491.mat';
%name_matfile = '2017-06-21_11:08_MLP_5lay_13fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.504testacc.mat';
%name_matfile = '2017-06-22_09:12_MLP_5lay_15fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.50testacc.mat';
%name_matfile = '2017-06-23_09:09_MLP_5lay_17fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.520testacc.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_tanh_2017-06-15_19:46_0.564.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_2048nodes_135class_20eps_relu_2017-06-16_19:12_0.600.mat'
%name_matfile = 'MLP_5lay_11fs_39coeffs_2048nodes_135class_20eps_selu_2017-06-14_18:44_0.575.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_sigmoid_2017-06-17_20:36_0.604.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_selu_2017-06-16_21:00_0.585.mat'
%name_matfile = 'MLP_5lay_9fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_2017-06-19_14:52_0.46.mat'
%name_matfile = '2017-06-21_11:08_MLP_5lay_13fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.504testacc.mat'
%name_matfile = '2017-06-22_09:12_MLP_5lay_15fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.50testacc.mat'
%name_matfile = '2017-06-23_09:09_MLP_5lay_17fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.520testacc.mat'
%name_matfile = '2017-06-25_19:25_MLP_5lay_21fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.536testacc.mat'
%name_matfile = '2017-06-29_13:12_MLP_5lay_11fs_257coeffs_1024nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.492testacc.mat'
%name_matfile = '2017-06-30_22:32_MLP_5lay_11fs_257coeffs_2048nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.504testacc.mat'
%name_matfile = '2017-06-30_02:42_MLP_5lay_11fs_257coeffs_1536nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.499testacc.mat'
%name_matfile = '2017-07-02_00:41_MLP_5lay_11fs_257coeffs_3072nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.497testacc.mat'

%to do:
% name_matfile = '2017-07-07_09:06_MLP_3lay_11fs_39coeffs_1024nodes_135class_20eps_relu_no_bnorm_no_lnorm_0.593testacc.mat'
%name_matfile = ''


%********************************************************
%call neural net:
%********************************************************
% path_matfile = '/data/rvg_new/nn_matfile/';
% net_name = strcat(path_matfile,name_matfile);
net_name = 'MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_relu_2017-06-15_19:46_0.601.mat'
%********************************************************
%create word hmms from dictionary:
%********************************************************
number_of_frames = 11; % define number of vectors as input to neural net
% refw = create_word_hmms_from_phonem_eri();
load('eri_word.mat')
%hmm_file = '/data/eri_german/config_files/commands_mono.hmm';
%********************************************************
%start recognition:
%******************************************************** 
% MFCC based recognition                      name_matfile(1:16)
name_type = 'eri_';
%save('eri_word.mat', 'refw','name_type');
% anal_recog_nn_liste('/data/eri_german/lists/commands_8khz.waves', strcat(fileparts(tmp.Filename),'/',name_type,net_name(1:16)), 8000, 4, 'MFCC_E_D_A', ...
% refw, '/data/eri_german/config_files/command_all_mono.syn', 'sil', net_name, number_of_frames);

% DFT based recognition
%anal_recog_nn_liste('/data/eri_german/lists/commands_16khz.waves', '/data/eri_german/rec_results_nn', 16000, 7, 'FBANK', ...
%refw, '/data/eri_german/config_files/command_all_mono.syn', 'sil', net_name, number_of_frames);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% anal_recog_liste
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sprachfile =  'commands_8khz.waves';
destpath = strcat(fileparts(tmp.Filename),'/',name_type,net_name(1:16));
fs = 8000;
vala = 4;
einstlg =  'MFCC_E_D_A';
hmmfile = refw;
synfile = 'command_all_mono.syn';
pausename = 'sil'; 
net_name = net_name; 
nr_frames = number_of_frames;

%  needs parameters:
%                   - wave list file
%                   - path to store the results
%                   - sampling frequency
%                   - analysis mode (=1 for ETSI-1  =2 for HGH  =3 for ETSI-2
%                     =4 for robust feature extraction including
%                     cepstro-temporal filtering
%                     =5 for FDLP processing 
%                     =6 for HGH_NR plus LDA features )
%                     =7 for simple DFT analysis
%                   - type of acoustic parameters as string (e.g.
%                     'MFCC_E_D_A')
%                   - name of list file containing HMMs
%                   - name of syntax file
%                   - name of pause model (usually 'w_sil')
%                   - adaptation flag (=0 no adaption  =1 adaptation  =5 adaptation of noisy HMMs)
%
% function anal_recog_nn_liste(sprachfile, destpath, fs, vala, einstlg, hmmfile, ...
%     synfile, pausename, net_name, nr_frames)  
    
ncep = 13;
if (vala == 2)
	nmel = 24; % HGH analysis    
elseif (vala == 3)    
	nmel = 23; % ETSI analysis
elseif  (vala == 4) % robust feature extraction    
    nmel = 24;
    % define init for noise reduction
    init.over_est_factor = 1;
    init.sr              = fs;
    init.cep_analysis    = 1; % =1 ==> creates a pattern file for recognition,
                              % =0 ==> creates a noise reduced time signal 
    init.string          = einstlg;
    init.algorithm       = 'lsa'; 
    init.cep_smooth      = true;
    init.framelength     = 200;
    init.frameshift      = 80;
    init.anal_mode       = 'training'; % use all frames
    %init.anal_mode       = 'test'; % use only frames 5 to end-5 
elseif (vala == 7) %
    nmel = 24; % NOT used
    initdft.use_preemp = true; 
    initdft.fs = fs;
    if (fs == 8000)
        initdft.framelength  = 200;
        initdft.frameshift   = 80;
        initdft.noverlap     = initdft.framelength - initdft.frameshift;
        initdft.nfft         = 256;
        initdft.bpre         = [ 1 -0.95 ];
    elseif (fs == 16000)
        initdft.framelength  = 400;
        initdft.frameshift   = 160;
        initdft.noverlap     = initdft.framelength - initdft.frameshift;
        initdft.nfft         = 512;
        initdft.bpre         = [ 0.0114 -0.0037 -0.0141 0.0009 0.0447 -0.0114 -0.1880 1.4774 -0.6140 ...
                             -0.4631 -0.2080 -0.0028 0.0365 0.0035 -0.01951 ];
    else
        fprintf(1, 'ERROR: Unknown sampling frequency: %d!\n', fs);
        return;
    end
    %load /data/rvg_new/config_files/mean_var_norm_dft257 % load mean and variance normalization parameters mm and vv
    load mean_var_norm_dft257
end


if exist(destpath, 'dir')
    dd = dir(destpath);
    if (length(dd) > 2)
        fprintf(1, 'WARNING: Directory %s is not empty!\n', destpath);
    end,       
else
    suc = mkdir(destpath);
    if ~suc
        fprintf(1, 'ERROR: cannot create directory %s!\n', destpath);
        return;
    end
end

if isstruct(hmmfile)
   ref = hmmfile;
else
   % load HMMS       
   [ref, error] = load_hmms(hmmfile, pausename);
   if (error)
      fprintf(1, 'ERROR: HMMs not available!\n');
      return;
   end,
end
% load mapping information
%load '/data/rvg_new/nn_output/nn_out_index.mat'
%ref.phonindex = cell(1, ref.no_of_refs);
%for ii=1:length(ref.name)
%    num = strmatch(ref.name{ii}, words, 'exact');
%    ref.phonindex{ii} = phonindex{num};
%end

% load syntax
[syn, ~, error] = load_syn(synfile, ref);
if (error)
    fprintf(1, 'ERROR: Syntax not available!\n');
    return;
end,

%addpath('/home/hirsch/work/projects/neural_net/scripts');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load neural net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%net_name = '/data/rvg_new/nn_matfile/net_in273_lay5_nodes2048_out135.mat';
%net_name = '/data/rvg_new/nn_matfile/MLP_5layer_2017-05-24_13:36_0.579.mat';
%net_name = '/data/rvg_new/nn_matfile/MLP_5layer_2017-06-02_19:42_0.557.mat';
[net, error] = load_tensor_net(net_name);
if (error)
    fprintf(1, 'ERROR: Neural net (file %s) not available!\n', net_name);
    return;
end,
fprintf(1,'Applying net with a total of %d layers and %d nodes for all layers besides the first one\n', length(net), size(net{2}.weights,1)); 


% preset filter weighting coefficients to 1
% and modification of log(energy) to 0
nest.h_f = ones(1, nmel);
nest.enerdiff = 0.;
nest.t60 = -1;

% open list file
fid = fopen(sprachfile, 'r');
linestr = fgetl(fid);
%linestr = '/data/rvg_new/speech/16khz/B_237/st1d0170_237.raw';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MFCC analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loop over all speech files
while ( linestr ~= -1)
    if (exist(linestr, 'file') == 2)
        x = loadshort(linestr); % load speech samples
    else
        fprintf(1, 'File %s does not exist!\n', linestr);
        linestr = fgetl(fid);
        continue;
    end,
    if (~isempty(x))
        fprintf(1, 'Analyzing and recognizing file %s ...\n', linestr);
        % perform feature extraction
        if (vala == 1) % ETSI-1
            mfcc = anal_etsi1(x, fs, einstlg);
        elseif (vala == 2)  % HGH
            [mfcc0, ~, noise_spec, nest] = anal_hgh(x, fs, einstlg, adapt, syn.timeout, nest);
            if (strcmp(einstlg, 'MFCC_E_D_A_0'))
                mfcc = [mfcc0(:,1:ncep-1) mfcc0(:,ncep+1:2*ncep) mfcc0(:,2*ncep+2:3*ncep+1) mfcc0(:,3*ncep+3)];
            else
                mfcc = mfcc0;
                clear mfcc0;
            end,
        elseif (vala == 3) % ETSI-2
            if (adapt && (strcmp(einstlg, 'MFCC_E&C0_D_A_0') == 0))
                fprintf('Can perform recognition with adaptation in combination with ETSI-2 analysis only with features of type MFCC_E&C0_D_A_0!\n');
                linestr = fgetl(fid);
                continue;
            end,
            mfcc0 = anal_etsi2_mod(x, fs, einstlg);
            if (strcmp(einstlg, 'MFCC_E&C0_D_A_0'))
                mfcc = [mfcc0(:,1:ncep-1) mfcc0(:,ncep+1:2*ncep) mfcc0(:,2*ncep+2:3*ncep+1) mfcc0(:,3*ncep+3)];
                einstlg = 'MFCC_E_D_A_0';
            else
                mfcc = mfcc0;
                clear mfcc0;
           end
        elseif (vala == 4)  % analysis with noise reduction
            [~, mfcc0] = anal_noise_reduction_mod(x, init);
            %[~, mfcc0, noise_spec, nest] = anal_noise_reduction_adapt(x, init, nest);
            if (strcmp(einstlg, 'MFCC_E_D_A_0'))
                mfcc = [mfcc0(:,1:ncep-1) mfcc0(:,ncep+1:2*ncep) mfcc0(:,2*ncep+2:3*ncep+1) mfcc0(:,3*ncep+3)];
            else
                mfcc = mfcc0;
                clear mfcc0;
            end,
        elseif (vala == 5)  % analysis with noise reduction
            mfcc = anal_fdlp(x, init)'; % code of anal_fdlp inside this m file
                 
        elseif (vala == 6)  % analysis with HGH_NR
            [~, mfcc, ~, ~, melspec] = anal_noise_reduction_mod(x, init);
        elseif (vala == 7) % plain DFT analysis
            %mfcc = spec_anal(x, initdft);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %call spec function
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
%function [spec] = spec_anal(sig, init)

sig = x;
init = initdft;

%---------------------------------------------------------------------
%--- FREQUENCY DOMAIN TRANSFORM
%---------------------------------------------------------------------
num_samples = length(sig);
num_frames  = fix((num_samples-init.noverlap)/init.frameshift);
specsize    = init.nfft/2+1;
win         = hamming(init.framelength);
%norm_win    = init.framelength/sum(win);

%---------------------------------------------------------------------
%--- optional PREEMPHASIS
%---------------------------------------------------------------------
if init.use_preemp == true
    [sig, z] = filter(init.bpre, 1, sig);
    if (init.fs > 8000)
        gd = (length(init.bpre)-1)/2;
        sig = [sig(gd+1:end); z(1:gd)];
    end   
end

spec     = zeros(num_frames, specsize);
%spec_noi = zeros(num_frames, specsize);
%nest_hgh.noise = 0;
for frame_idx = 1:num_frames,
    start_idx = (frame_idx-1)*init.frameshift+1;
    stop_idx  = (frame_idx-1)*init.frameshift+init.framelength;
    noisy_frame = win .* sig(start_idx:stop_idx);
    noisy_transf = fft(noisy_frame, init.nfft);
    spec(frame_idx,:) = log(max(abs(noisy_transf(1:specsize)), 1));
    %nest_hgh = est_noise_spec(abs(spec(frame_idx,:)), nest_hgh, frame_idx);
    %spec_noi(frame_idx,:) = init.noise_overestimation * nest_hgh.noise;
end % frame_idx

%end
            
  mfcc = spec;          
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %end spec function
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            nvecin = size(mfcc, 1);
            % mean and variance normalization
            mfcc = (mfcc + repmat(mm, nvecin, 1)) .* repmat(vv, nvecin, 1);
            % mean and variance normalization per utterance
            %mfcc = (mfcc - repmat(mean(mfcc), nvecin, 1)) ./ repmat(sqrt(var(mfcc)), nvecin, 1);
            %inscale = (in + repmat(nnet{1}.addshift, nvecin, 1)) .* repmat(nnet{1}.rescale, nvecin, 1);
        end,
        no_of_frames = size(mfcc,1);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init viterbi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

        % perform recognition without adaptation
        dis = viterbi_initset(ref, syn);
        dis_xxxx1 = dis;
        dis = viterbi_reset(dis, ref, syn);
        dis_xxxx2 = dis;

        %bp = best_path_init( ref, syn, no_of_frames);

        bestref = zeros(syn.num_of_nodes, no_of_frames);
        fromframe = zeros(syn.num_of_nodes, no_of_frames);
        %local_prob = cell(1,no_of_frames); 
        nf = (nr_frames-1) / 2;
        
        %padded data
        mfcc7 = [repmat(mfcc(1,:),nf,1); mfcc; repmat(mfcc(end,:),nf,1)];
        
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
 %start viterbi recognition loop for one file
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 for count = 1:no_of_frames
       vec = mfcc(count,:);
                 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %start viterbi sync
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %dis = viterbi_syn_calc_nn(vec, dis, ref, syn, net, mfcc7, nr_frames, count);            
%function dis = viterbi_syn_calc_nn (vec, dis, ref, syn, net, mfcc7, nr_frames, count)

%compare actual vector "vec" with all states of all HMMs
% dis = calc_local_prob_opt(vec, ref, dis); 
% call of calc_local_prob_opt substituted by calling a mex function
% err = calc_local_prob_c(ref, vec, dis);
% if (err == 1)
%     error('ERROR in Mex function calc_local_prob_c!!!\n');
% end


if isfield(net{1}, 'layertype') && strcmp(net{1}.layertype, 'conv')
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %load cnn
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %[out, err] = apply_convnet(mfcc7(count:count+nr_frames-1,:)', net);
    
    in = mfcc7(count:count+nr_frames-1,:)';
    nnet = net;
    
%function [out, err] = apply_convnet(in, nnet)
%

% mystic numbers for SELU
ALPHA  = 1.6732632423543772848170429916717;
LAMBDA = 1.0507009873554804934193349852946;

err = 0;
out = [];
nlayers = length(nnet);
outl = cell(1, nlayers);
[nspec, nf_kernel, dum, nkernel] = size(nnet{1}.weights);
[nlen, nframes] = size(in);
if (nframes ~= nf_kernel)
    fprintf(1, 'ERROR: number of frames (%d) does not correspond with kernel size (%d)!\n', nframes, nf_kernel);
    return;
end
%ind = 1;
oo = zeros(1, nkernel*(nlen-nspec+1));
for ii=1:nkernel
    for jj=1:nlen-nspec+1
        oo((jj-1)*nkernel +ii) = sum(sum(in(jj:jj+nspec-1,:).*nnet{1}.weights(:,:,1,ii))) + nnet{1}.bias(1,ii);
        %ind = ind + 1;
    end
end

lenvecin = length(oo);
nvecin = 1;
[nin, nout] = size(nnet{2}.weights);
if (nin ~= lenvecin)
    fprintf(1, 'Size of 2nd layer NOT adequate for output of Convolutional layer (%d != %d)!\n', nin, lenvecin);
    err = 1;
    return;
end
% activation function of 1st layer
if strcmp(nnet{1}.nonlin, 'tanh')
    outl{1} = tanh(oo); % nonlinear weighting with tanh
elseif strcmp(nnet{1}.nonlin, 'sigmoid')
    outl{1} = 1./(exp(-oo)+1);
elseif strcmp(nnet{1}.nonlin, 'relu')
    outl{1} = max(oo, 0);
elseif strcmp(nnet{1}.nonlin, 'selu')
    % calculate values
    %negative = LAMBDA * ((ALPHA * exp(oo)) - ALPHA);
    negative = LAMBDA * ALPHA * (exp(oo) - 1);
    positive = LAMBDA * oo;
    negative (oo > 0.0) = 0;
    positive (oo <= 0.0) = 0;
    % result
    outl{1} = positive + negative; 
elseif strcmp(nnet{1}.nonlin, 'none')
    outl{1} = oo;
end

for ii=2:nlayers    
    [nin, nout] = size(nnet{ii}.weights);
    if (nin ~= lenvecin)
        fprintf(1, 'Size of %d. layer NOT adequate (%d != %d)!\n', ii, nin, lenvecin);
        err = 1;
        return;
    end
    oo = (outl{ii-1} * nnet{ii}.weights + repmat(nnet{ii}.bias, nvecin, 1)); %/ (nin+1);
    outl{ii} = oo;
    if strcmp(nnet{ii}.nonlin, 'tanh')
        outl{ii} = tanh(oo); % nonlinear weighting with tanh
    elseif strcmp(nnet{1}.nonlin, 'sigmoid')
        outl{ii} = 1./(exp(-oo)+1);
    elseif strcmp(nnet{ii}.nonlin, 'relu')
        outl{ii} = max(oo, 0);
    elseif strcmp(nnet{ii}.nonlin, 'selu')
        negative = LAMBDA * ALPHA * (exp(oo) - 1);
        positive = LAMBDA * oo;
        negative (oo > 0.0) = 0;
        positive (oo <= 0.0) = 0;
        outl{ii} = positive + negative; 
    elseif strcmp(nnet{ii}.nonlin, 'softmax')
        outl{ii} = exp(outl{ii})/sum(exp(outl{ii}));
        %outl{ii} = bsxfun(@rdivide, exp(outl{ii}), sum(exp(outl{ii})));
    elseif strcmp(nnet{ii}.nonlin, 'none')
        outl{ii} = oo;
    end
    lenvecin = nout;
end

%out = exp(outl{nlayers})/sum(exp(outl{nlayers}));
out = outl{nlayers};
%end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%end cnn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    
else
    in = reshape(mfcc7(count:count+nr_frames-1,:)', 1, nr_frames*size(mfcc7,2));
    %in = reshape(mfcc7(count:count+6,:), 1, 7*size(mfcc7,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load mlp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[out, err] = apply_net(in, net);
    
    nnet = net;
    
    %function [out, err] = apply_net(in, nnet)
%

% mystic numbers for SELU
ALPHA  = 1.6732632423543772848170429916717;
LAMBDA = 1.0507009873554804934193349852946;

err = 0;
out = [];
nlayers = length(nnet);
outl = cell(1, nlayers);
[nvecin, lenvecin] = size(in); % number of vector components
[nin, nout] = size(nnet{1}.weights);
if (nin ~= lenvecin)
    fprintf(1, 'Size of 1st layer NOT adequate (%d != %d)!\n', nin, lenvecin);
    err = 1;
    return;
end
% add Offset and Rescale input parameters
%inscale = (in + repmat(nnet{1}.addshift, nvecin, 1)) .* repmat(nnet{1}.rescale, nvecin, 1);
inscale = in;
oo = (inscale * nnet{1}.weights + repmat(nnet{1}.bias, nvecin, 1));
if strcmp(nnet{1}.nonlin, 'tanh')
    outl{1} = tanh(oo); % nonlinear weighting with tanh
elseif strcmp(nnet{1}.nonlin, 'sigmoid')
    outl{1} = 1./(exp(-oo)+1);
elseif strcmp(nnet{1}.nonlin, 'relu')
    outl{1} = max(oo, 0);
elseif strcmp(nnet{1}.nonlin, 'selu')
    % calculate values
    %negative = LAMBDA * ((ALPHA * exp(oo)) - ALPHA);
    negative = LAMBDA * ALPHA * (exp(oo) - 1);
    positive = LAMBDA * oo;
    negative (oo > 0.0) = 0;
    positive (oo <= 0.0) = 0;
    % result
    outl{1} = positive + negative; 
elseif strcmp(nnet{1}.nonlin, 'none')
    outl{1} = oo;
end
for ii=2:nlayers
    lenvecin = nout;
    [nin, nout] = size(nnet{ii}.weights);
    if (nin ~= lenvecin)
        fprintf(1, 'Size of %d. layer NOT adequate (%d != %d)!\n', ii, nin, lenvecin);
        err = 1;
        return;
    end
    oo = (outl{ii-1} * nnet{ii}.weights + repmat(nnet{ii}.bias, nvecin, 1)); %/ (nin+1);
    outl{ii} = oo;
    if strcmp(nnet{ii}.nonlin, 'tanh')
        outl{ii} = tanh(oo); % nonlinear weighting with tanh
    elseif strcmp(nnet{1}.nonlin, 'sigmoid')
        outl{ii} = 1./(exp(-oo)+1);
    elseif strcmp(nnet{ii}.nonlin, 'relu')
        outl{ii} = max(oo, 0);
    elseif strcmp(nnet{ii}.nonlin, 'selu')
        negative = LAMBDA * ALPHA * (exp(oo) - 1);
        positive = LAMBDA * oo;
        negative (oo > 0.0) = 0;
        positive (oo <= 0.0) = 0;
        outl{ii} = positive + negative; 
    elseif strcmp(nnet{ii}.nonlin, 'softmax')
        outl{ii} = exp(outl{ii})/sum(exp(outl{ii}));
    elseif strcmp(nnet{ii}.nonlin, 'none')
        outl{ii} = oo;
    end    
end

%out = exp(outl{nlayers})/sum(exp(outl{nlayers}));
out = outl{nlayers};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%end mlp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
          

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%start actual viterbi for one softmax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%loop 65 times in case of eri
for ii=1:ref.no_of_refs
    oocheck = out(ref.phonindex{ii});
    oocheck2 = log(max(1e-6, out(ref.phonindex{ii})));
    dis.local_prob{ii} = log(max(1e-6, out(ref.phonindex{ii})));  
end
%load('st1d0388_395.mat');
%load('t7lc0019_089.mat');
if (err == 1)
    error('ERROR in neural net function!!!\n');
end

%work on all nodes in syntax
for node_ind = 1:syn.num_of_nodes,
    
    %work on each model ending in this node
    for ref_ind = 1:syn.no_of_refs(node_ind),
        synentrycheck = syn.entry_nodes{node_ind}(ref_ind);
        
        % Activate only references starting at node one
        if (count == 1) && (syn.entry_nodes{node_ind}(ref_ind) == 1)
            % next 2 lines only for HMMs with NO skip over the first state
            %dis.sumprob{node_ind}{ref_ind}(2) = dis.local_prob{syn.ref_index{node_ind}(ref_ind)}(1);
            %dis.from_frame{node_ind}{ref_ind}(2) = -1;
            %fprintf(1,'Reference: %s prob: %f\n', ref.name{syn.ref_index{node_ind}(ref_ind)}, dis.sumprob{node_ind}{ref_ind}(2));
            %continue;
            % following loop to include also skips over the first state
            for i=2:dis.maxskip(syn.ref_index{node_ind}(ref_ind)) + 1
                dis_maxcheck = dis.maxskip(syn.ref_index{node_ind}(ref_ind)) + 1; %for check only
                dis.sumprob{node_ind}{ref_ind}(i) = dis.local_prob{syn.ref_index{node_ind}(ref_ind)}(i-1) + ...
                    ref.transp{syn.ref_index{node_ind}(ref_ind)}(1,i);
                refAtranscheck = ref.transp{syn.ref_index{node_ind}(ref_ind)}(1,i);
                dis.from_frame{node_ind}{ref_ind}(i) = -1;
            end
        end
        if ( count == 1 )
            continue;    %skip the rest of the loop
        end
        
        %only executed for count bigger than 1 (count ist 193 je ein
        %softmax)
        zzzz=5;
        
        % calculation of accumulated probability for all states > 1
        oldframe = dis.oldframe{node_ind}{ref_ind};
        local_prob = dis.local_prob{syn.ref_index{node_ind}(ref_ind)};
        oldsum = dis.oldsum {node_ind}{ref_ind};
        transp = ref.transp{syn.ref_index{node_ind}(ref_ind)};
        maxskip = dis.maxskip;
        minskip = dis.minskip;
        numstates = ref.numstates;
        for state_ind = 2:ref.numstates(syn.ref_index{node_ind}(1,ref_ind))+1
            numstate_check = ref.numstates(syn.ref_index{node_ind}(1,ref_ind))+1;
            
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%start function max_search_opt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%            dmax = max_search(dis, ref, node_ind, syn.ref_index{node_ind}(1,ref_ind), state_ind);
%            [dmax,max_ind] = max_search_opt(oldsum, transp, numstates, maxskip, minskip, node_ind, ref_ind,...
%                syn.ref_index{node_ind}(ref_ind), state_ind);
%            
%            function [dmax,max_ind] = max_search_opt (oldsum, transp, numstates, mxskip, mnskip, node_ind,...
%     ref_ind, refliste_ind, state_ind)

mxskip = maxskip;
mnskip = minskip;
refliste_ind = syn.ref_index{node_ind}(ref_ind);

%node_ind
%ref_ind
%j = state_ind;
minskip_x = mnskip(1, refliste_ind);
maxskip_x = mxskip(1, refliste_ind);
dmax = oldsum(state_ind) + transp(state_ind, state_ind);
oldsumcheck = oldsum(state_ind);
transpcheck = transp(state_ind, state_ind);
max_ind =0;

j = state_ind + 1;
if minskip_x > 0
    while ((j <= numstates(refliste_ind)+1) && (minskip_x > 0))
        [dmax, ind] = max([dmax, dis.oldsum{node_ind}{ref_ind}(j) + ref.transp{refliste_ind}(j, state_ind)]);
        if ind == 2
            max_ind = state_ind -j;
        end
        j = j + 1;
        minskip_x = minskip_x - 1;
    end,
end,
j = state_ind - 1;
while ((j >= 1) && (maxskip_x > 0))
    oldsumcheck2 = oldsum(j);
    transpcheck2 = transp(j, state_ind);
    new = oldsum(j) + transp(j, state_ind);
%    [dmax, ind] = max([dmax, dis.oldsum{node_ind}{ref_ind}(j) + ref.transp{refliste_ind}(j, state_ind)]);
%    if ind == 2
    if new > dmax
        max_ind = state_ind -j;
        dmax = new;
    end
    j = j - 1;
    maxskip_x = maxskip_x - 1;
end
           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%end function max_search_opt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
olf2check = oldframe(state_ind-max_ind);
local2check = local_prob(state_ind-1)+dmax;

            dis.from_frame{node_ind}{ref_ind}(state_ind) = oldframe(state_ind-max_ind);           
            dis.sumprob{node_ind}{ref_ind}(state_ind) = local_prob(state_ind-1)+dmax;
        end,
    end,
    %
end,


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [ dis ] = calc_node_prob( dis,ref,syn,count )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dis = calc_node_prob(dis, ref, syn, count);

% function [ dis ] = calc_node_prob( dis,ref,syn,count )
%CALC_NODE_PROB Summary of this function goes here
%  Detailed explanation goes here


%look at all nodes of the syntax
for node_ind = 1:syn.num_of_nodes
    %fprintf(1,'working on node %d ...\n', node_ind);
    dis.node_prob(node_ind) = -Inf;
    dis.best_ref(node_ind) = -1;
    dis.best_from (node_ind) = -1;    
    
    % look at all states of all models ending in this node
%     prob_high = - Inf; % highest probability of all HMMs at all states exiting this model 
    for ref_ind = 1:syn.no_of_refs(node_ind)
                       
        state_ind = ref.numstates(syn.ref_index{node_ind}(1,ref_ind)) + 1;
        skip = dis.maxskip(1,syn.ref_index{node_ind}(1,ref_ind));
        
        while (state_ind > 1) && (skip > 0)
            dum = dis.sumprob{node_ind}{ref_ind}(state_ind);
            if ( dum > -Inf )
                dum = dum + ref.transp{syn.ref_index{node_ind}(ref_ind)}(state_ind, ...
                    ref.numstates(syn.ref_index{node_ind}(ref_ind))+2);
                
                if dum > dis.node_prob(node_ind)
                    dis.node_prob(node_ind) = dum;
                    dis.best_ref(node_ind) = ref_ind;
                    dis.best_from(node_ind) = dis.from_frame{node_ind}{ref_ind}(state_ind);
                end,
            end,
            state_ind = state_ind - 1;
            skip = skip - 1;
        end,
    end,
end,

for node_ind = 1:syn.num_of_nodes
    for ref_ind = 1:syn.no_of_refs(node_ind)
        synentrycheck = syn.entry_nodes{node_ind}(ref_ind);
        if (dis.node_prob(syn.entry_nodes{node_ind}(ref_ind)) > -Inf )
            dis.sumprob{node_ind}{ref_ind}(1) = dis.node_prob(syn.entry_nodes{node_ind}(ref_ind));
            dis.from_frame{node_ind}{ref_ind}(1) = count;
        else
            dis.sumprob{node_ind}{ref_ind}(1) = - Inf;
            dis.from_frame{node_ind}{ref_ind}(1) = 0;
        end
        %fprintf(1,'assigning %f to ref %d of node %d ...\n', dis.sumprob{node_ind}{ref_ind}(1), node_ind, ref_ind);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end calc_node_prob
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for node_ind = 1:syn.num_of_nodes
    for ref_ind = 1:syn.no_of_refs(node_ind)
        dis.oldsum{node_ind}{ref_ind} = dis.sumprob{node_ind}{ref_ind};
        dis.oldframe{node_ind}{ref_ind} = dis.from_frame{node_ind}{ref_ind};
        
    end,
end,
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %end viterbi sync
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
 
 
            bestref(:,count) = dis.best_ref'; %transpose
            fromframe(:,count) = dis.best_from'; %transpose
            %local_prob{count} = dis.local_prob;
            %bp = best_path_copy( bp, ref, syn, dis, count );
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %end of loop for one file
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

%here already total probability:
max(dis.node_prob) 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %backtrack viterbi sync (for recognizing the sequence , the word)
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

        %res = backtrack_viterbi_syn ( ref, syn, dis, bestref, fromframe, no_of_frames );
        
    % The function backtrack_viterbi_syn performs backtracking based on the
% information given in bestref and fromframe
%function res = backtrack_viterbi_syn ( ref, syn, dis, bestref, fromframe, no_of_frames )

score = -Inf;
best_node = -1;

% find node with highest probability
if (syn.num_endnodes == -1)  % No final nodes defined in the syntax look at all nodes
    for node_ind = 1:syn.num_of_nodes         
        if dis.node_prob(node_ind) > score    
            score = dis.node_prob(node_ind); 
            best_node = node_ind;
        end
    end
else  % looking only at the final nodes as defined in the syntax
    for i=1:syn.num_endnodes
        node_ind = syn.endnodes(i);
        if dis.node_prob(node_ind) > score   
            score = dis.node_prob(node_ind);
            best_node = node_ind;
        end
    end
end,        

if (best_node == -1)
    res.num_of_words = 0;
    fprintf('WARNING: NO node with finite probability found!');
    return;
end,
% find best reference in this node
% prob_high = - Inf; % highest probability of all HMMs at all states exiting this model 
% best_ref = -1;
% best_state = -1;
% for ref_ind = 1:syn.no_of_refs(best_node),
%                        
%     state_ind = ref.numstates(syn.ref_index{best_node}(1,ref_ind))+1;
%     skip = dis.maxskip(1,syn.ref_index{best_node}(1,ref_ind));
%         
%     while (state_ind > 1) && (skip > 0)
%             dum = dis.sumprob{best_node}{ref_ind}(state_ind);
%             if (dum > prob_high)
%                 best_ref = ref_ind;
%                 best_state = state_ind;
%                 prob_high = dum;
%             end,
%             state_ind = state_ind - 1;
%             skip = skip - 1;
%    end,
% end,
% 
% bestref = [bestref zeros(syn.num_of_nodes,1)];
% fromframe = [fromframe zeros(syn.num_of_nodes,1)];
% i = no_of_frames+1;
% bestref(best_node, i) = best_ref;
% fromframe(best_node, i) = dis.from_frame{best_node}{best_ref}(state_ind);

i = no_of_frames;  % start backtracking at the last frame 
cc = 0;            % counter for the number of recognized HMMs
j = best_node;
ind = -1;

while i > 1
    oldind = ind;
    oldi = i;
    oldj = j;
    %fprintf(1, 'best reference at node %d and at frame index %d: %d\n', oldj, oldi, bestref(oldj,oldi));
    j = syn.entry_nodes{oldj}(bestref(oldj,oldi));
    i = fromframe( oldj, oldi );
    ind = syn.ref_index{oldj}(bestref(oldj,oldi));
    
    % Do not count the pause model in case it occurs consecutively
    if ( (oldind > 0) && (strcmp(ref.name(ind), ref.silstr ) == 1 )...
       && (strcmp(ref.name(ind), ref.name(oldind)) == 1))
       continue;
    end
    
    res.word_ind(1,cc+1) = ind;
    res.time_ind(1,cc+1) = oldi;
    
    cc = cc + 1;
end
res.word_ind=res.word_ind(cc:-1:1);
res.time_ind=res.time_ind(cc:-1:1);
res.num_of_words = cc;

% screen output
fprintf(1, 'Recognized HMMs:');
for i=1:res.num_of_words
    fprintf(1, '  %s' ,ref.name{res.word_ind(i)});
end
fprintf(1, '\n');    
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  end backtrack viterbi sync
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
              

% if (vala == 6)
%        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %vala 6 is not needed!!! 
% %call function: but not needed
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %res = map2singleres(res, ref); % eliminate left models
%             
%             
%             res_tot = res;
%             ref_lr = ref;
%             
% %             function res = map2singleres(res_tot, ref_lr)
% res.num_of_words = 0;
% for nr=1:res_tot.num_of_words
%     if strcmp(ref_lr.name{res_tot.word_ind(1,nr)}, ref_lr.silstr)
%         res.num_of_words = res.num_of_words + 1;
%         res.word_ind(1,res.num_of_words) = res_tot.word_ind(1,nr);
%         res.time_ind(1,res.num_of_words) = res_tot.time_ind(1,nr);        
%     elseif strcmp(ref_lr.name{res_tot.word_ind(1,nr)}(5), 'r')
%         res.num_of_words = res.num_of_words + 1;
%         res.word_ind(1,res.num_of_words) = res_tot.word_ind(1,nr);
%         res.time_ind(1,res.num_of_words) = res_tot.time_ind(1,nr);
%     end
% end
% % end          
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% final statistics and output file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        %bp = best_path_calc( bp, ref, syn, dis, no_of_frames );

        fprintf('Total probability: %f\n', max(dis.node_prob));
        % create name of output file
        [~,n,e] = fileparts(linestr);
        if ( strcmp(e, '.srt') || strcmp(e, '.raw') )
            outname = fullfile(destpath, [n '.rec']);
        else
            e = strcat(e, '.rec');
            outname = fullfile(destpath, [n e]);
        end,
        
        % sampling frequency, window length and window shift needed
        %  for time labelling in result file
        rec.s_rate = fs;
        rec.ws = 1e-2 * rec.s_rate; % 10 ms window shift
        rec.wl = 200;  % window length

        create_label(outname, res, ref, syn, rec );

    else
        fprintf(1, 'File %s has zero length!\n', linestr);
    end,
    linestr = fgetl(fid); %read in new file
end,

fclose(fid);
% end



disp('================================')
disp('finished HMM decoding')
disp('================================')
endtime = toc/60
disp(['total decoding time: ',num2str(endtime/3600),char(10) ])

















