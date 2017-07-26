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

addpath('/packages/speech_recognition/matlab');
wave_list = '/data/rvg_new/lists/digits.waves'; % 8 kHz MFCCs
path_matfile = '/data/rvg_new/nn_matfile/';

addpath('/packages/speech_recognition/matlab');
%********************************************************
%insert neural net names:
%********************************************************

%wave_list = '/data/rvg_new/lists/digits_16k.waves'; % 16 kHz DFT based neural nets
%wave_list = '/data/rvg_new/lists/ttt.waves';
%name_matfile = 'MLP_5layer_2017-05-10_16:39_0.497.mat';
%name_ma%wave_list = '/data/rvg_new/lists/digits_16k.waves'; % 16 kHz DFT based neural nets
%wave_list = '/data/rvg_new/lists/ttt.waves';
%name_matfile = 'MLP_5layer_2017-05-10_16:39_0.497.mat';
%name_matfile = 'MLP_5layer_2017-05-20_15:44_0.560.mat';
%name_matfile = 'MLP_5layer_bnormentry_fft_2017-06-09_12:56_0.398.mat';
%name_matfile = 'MLP_5lay_7fs_257coeffs_512nodes_135class_20eps_relu2017-06-19_14:50_0.449.mat';
%name_matfile = 'MLP_5lay_11fs_257coeffs_512nodes_135class_20eps_relu2017-06-20_08:58_0.491.mat';
%name_matfile = '2017-06-21_11:08_MLP_5lay_13fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.504testacc.mat';
%name_matfile = '2017-06-22_09:12_MLP_5lay_15fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.50testacc.mat';
%name_matfile = '2017-06-23_09:09_MLP_5lay_17fs_257coeffs_512nodes_135class_20eps_relu_bnorm_entry_no_lnorm_0.520testacc.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_tanh_2017-06-15_19:46_0.564.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_2048nodes_135class_20eps_relu_2017-06-16_19:12_0.600.mat';

%to do:
name_matfile = 'MLP_5lay_11fs_39coeffs_2048nodes_135class_20eps_selu_2017-06-14_18:44_0.575.mat';
%name_matfile = 'MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_sigmoid_2017-06-17_20:36_0.604.mat';
net_name = strcat(path_matfile,name_matfile);

%********************************************************
%create word hmms from dictionary:
%********************************************************
number_of_frames = 11; % define number of vectors as input to neural net
refw = create_word_hmms_from_phonem_eri();
%hmm_file = '/data/eri_german/config_files/commands_mono.hmm';

%********************************************************
%start recognition:
%******************************************************** 
% MFCC based recognition
name_type = 'eri_';
anal_recog_nn_liste('/data/eri_german/lists/commands_8khz.waves', strcat(fileparts(tmp.Filename),'/',name_type,name_matfile), 8000, 4, 'MFCC_E_D_A', ...
refw, '/data/eri_german/config_files/command_all_mono.syn', 'sil', net_name, number_of_frames);

% DFT based recognition
%anal_recog_nn_liste('/data/eri_german/lists/commands_16khz.waves', '/data/eri_german/rec_results_nn', 16000, 7, 'FBANK', ...
%refw, '/data/eri_german/config_files/command_all_mono.syn', 'sil', net_name, number_of_frames);

disp('================================')
disp('finished HMM decoding')
disp('================================')
endtime = toc/60
disp(['total decoding time: ',num2str(endtime/3600),char(10) ])

















