% anal_recog_liste
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
function anal_recog_nn_liste(sprachfile, destpath, fs, vala, einstlg, hmmfile, ...
    synfile, pausename, net_name, nr_frames)  
    
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
    load /data/rvg_new/config_files/mean_var_norm_dft257 % load mean and variance normalization parameters mm and vv
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

addpath('/home/hirsch/work/projects/neural_net/scripts');
% load neural net
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
            mfcc = spec_anal(x, initdft);
            nvecin = size(mfcc, 1);
            % mean and variance normalization
            mfcc = (mfcc + repmat(mm, nvecin, 1)) .* repmat(vv, nvecin, 1);
            % mean and variance normalization per utterance
            %mfcc = (mfcc - repmat(mean(mfcc), nvecin, 1)) ./ repmat(sqrt(var(mfcc)), nvecin, 1);
            %inscale = (in + repmat(nnet{1}.addshift, nvecin, 1)) .* repmat(nnet{1}.rescale, nvecin, 1);
        end,
        no_of_frames = size(mfcc,1);
        

        % perform recognition without adaptation
        dis = viterbi_initset(ref, syn);
        dis = viterbi_reset(dis, ref, syn);

        %bp = best_path_init( ref, syn, no_of_frames);

        bestref = zeros(syn.num_of_nodes, no_of_frames);
        fromframe = zeros(syn.num_of_nodes, no_of_frames);
        %local_prob = cell(1,no_of_frames); 
        nf = (nr_frames-1) / 2;
        mfcc7 = [repmat(mfcc(1,:),nf,1); mfcc; repmat(mfcc(end,:),nf,1)];
        for count = 1:no_of_frames
            vec = mfcc(count,:);
            dis = viterbi_syn_calc_nn(vec, dis, ref, syn, net, mfcc7, nr_frames, count);
            bestref(:,count) = dis.best_ref';
            fromframe(:,count) = dis.best_from';
            %local_prob{count} = dis.local_prob;
            %bp = best_path_copy( bp, ref, syn, dis, count );
        end

        res = backtrack_viterbi_syn ( ref, syn, dis, bestref, fromframe, no_of_frames );
        if (vala == 6)
            res = map2singleres(res, ref); % eliminate left models
        end
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
    linestr = fgetl(fid);
end,

fclose(fid);
end

function dis = viterbi_syn_calc_nn (vec, dis, ref, syn, net, mfcc7, nr_frames, count)

%compare actual vector "vec" with all states of all HMMs
% dis = calc_local_prob_opt(vec, ref, dis); 
% call of calc_local_prob_opt substituted by calling a mex function
% err = calc_local_prob_c(ref, vec, dis);
% if (err == 1)
%     error('ERROR in Mex function calc_local_prob_c!!!\n');
% end
in = reshape(mfcc7(count:count+nr_frames-1,:)', 1, nr_frames*size(mfcc7,2));
%in = reshape(mfcc7(count:count+6,:), 1, 7*size(mfcc7,2));
[out, err] = apply_net(in, net);
for ii=1:ref.no_of_refs
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
        
        % Activate only references starting at node one
        if (count == 1) && (syn.entry_nodes{node_ind}(ref_ind) == 1)
            % next 2 lines only for HMMs with NO skip over the first state
            %dis.sumprob{node_ind}{ref_ind}(2) = dis.local_prob{syn.ref_index{node_ind}(ref_ind)}(1);
            %dis.from_frame{node_ind}{ref_ind}(2) = -1;
            %fprintf(1,'Reference: %s prob: %f\n', ref.name{syn.ref_index{node_ind}(ref_ind)}, dis.sumprob{node_ind}{ref_ind}(2));
            %continue;
            % following loop to include also skips over the first state
            for i=2:dis.maxskip(syn.ref_index{node_ind}(ref_ind)) + 1
                dis.sumprob{node_ind}{ref_ind}(i) = dis.local_prob{syn.ref_index{node_ind}(ref_ind)}(i-1) + ...
                    ref.transp{syn.ref_index{node_ind}(ref_ind)}(1,i);
                dis.from_frame{node_ind}{ref_ind}(i) = -1;
            end
        end
        if ( count == 1 )
            continue;
        end
        
        % calculation of accumulated probability for all states > 1
        oldframe = dis.oldframe{node_ind}{ref_ind};
        local_prob = dis.local_prob{syn.ref_index{node_ind}(ref_ind)};
        oldsum = dis.oldsum {node_ind}{ref_ind};
        transp = ref.transp{syn.ref_index{node_ind}(ref_ind)};
        maxskip = dis.maxskip;
        minskip = dis.minskip;
        numstates = ref.numstates;
        for state_ind = 2:ref.numstates(syn.ref_index{node_ind}(1,ref_ind))+1,
%            dmax = max_search(dis, ref, node_ind, syn.ref_index{node_ind}(1,ref_ind), state_ind);
           [dmax,max_ind] = max_search_opt(oldsum, transp, numstates, maxskip, minskip, node_ind, ref_ind,...
               syn.ref_index{node_ind}(ref_ind), state_ind);

            dis.from_frame{node_ind}{ref_ind}(state_ind) = oldframe(state_ind-max_ind);
           
            dis.sumprob{node_ind}{ref_ind}(state_ind) = local_prob(state_ind-1)+dmax;
        end,
    end,
end,

dis = calc_node_prob(dis, ref, syn, count);

for node_ind = 1:syn.num_of_nodes,
    for ref_ind = 1:syn.no_of_refs(node_ind),
        dis.oldsum{node_ind}{ref_ind} = dis.sumprob{node_ind}{ref_ind};
        dis.oldframe{node_ind}{ref_ind} = dis.from_frame{node_ind}{ref_ind};
        
    end,
end,
end

function [out, err] = apply_net(in, nnet)
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

function dis = viterbi_syn_calc_lda(vec, dis, ref, syn, count, melspec, lda_all, hmm_list)

% calculate emission probabilities for all states of all HMMs
dis = calc_local_prob_lda(vec, ref, dis, melspec, lda_all, hmm_list); 

% work on all nodes in syntax
for node_ind = 1:syn.num_of_nodes,
    
    %work on each model ending in this node
    for ref_ind = 1:syn.no_of_refs(node_ind),
        
        % Activate only references starting at node one
        if (count == 1) && (syn.entry_nodes{node_ind}(ref_ind) == 1)
            % following loop to include also skips over the first state
            for i=2:dis.maxskip+1
                dis.sumprob{node_ind}{ref_ind}(i) = dis.local_prob{syn.ref_index{node_ind}(ref_ind)}(i-1) + ...
                    ref.transp{syn.ref_index{node_ind}(ref_ind)}(1,i);
                dis.from_frame{node_ind}{ref_ind}(i) = -1;
            end
        end
        if ( count == 1 )
            continue;
        end
        
        % calculation of accumulated probability for all states > 1
        oldframe = dis.oldframe{node_ind}{ref_ind};
        local_prob = dis.local_prob{syn.ref_index{node_ind}(ref_ind)};
        oldsum = dis.oldsum {node_ind}{ref_ind};
        transp = ref.transp{syn.ref_index{node_ind}(ref_ind)};
        maxskip = dis.maxskip;
        minskip = dis.minskip;
        numstates = ref.numstates;
        for state_ind = 2:ref.numstates(syn.ref_index{node_ind}(1,ref_ind))+1,
           [dmax,max_ind] = max_search_opt(oldsum, transp, numstates, maxskip, minskip, node_ind, ref_ind,...
               syn.ref_index{node_ind}(ref_ind), state_ind);

            dis.from_frame{node_ind}{ref_ind}(state_ind) = oldframe(state_ind-max_ind);
           
            dis.sumprob{node_ind}{ref_ind}(state_ind) = local_prob(state_ind-1)+dmax;
        end,
    end,
end,

dis = calc_node_prob(dis, ref, syn, count);

for node_ind = 1:syn.num_of_nodes,
    for ref_ind = 1:syn.no_of_refs(node_ind),
        dis.oldsum{node_ind}{ref_ind} = dis.sumprob{node_ind}{ref_ind};
        dis.oldframe{node_ind}{ref_ind} = dis.from_frame{node_ind}{ref_ind};
        
    end,
end,

end

function [dis] = calc_local_prob_lda(vec, ref, dis, melspec, lda_all, hmmlist)

for ref_ind = 1:ref.no_of_refs,
    hmmname = ref.name{ref_ind};
    if strcmp(hmmname, ref.silstr)
        ind = find(strncmp('s_1_l_f', hmmlist, 7), 1);
    else
        ind = find(strncmp(hmmname, hmmlist, 7), 1);
    end
    %fprintf(1, 'Applying %d. LDA matrix for model %s in %d. segment between frames %d and %d\n', ind, hmmname, ...
    %    ik, startind, endind);
    ldamat = lda_all{ind};
    out = melspec * ldamat;
    lda_feat = [vec, out];

    for state_ind = 1:ref.numstates(ref_ind),
        sum2 = 0;
        means = ref.means{ref_ind}{state_ind};
        variances = ref.variances {ref_ind}{state_ind};
        weight = dis.weight{ref_ind}{state_ind};
        
        for mix_ind = 1:ref.nummixes{ref_ind}(state_ind),
                       
            diff = means(mix_ind,:) - lda_feat;
            sum1 = sum(diff .* diff ./ variances(mix_ind,:));
            sum1 = exp(-0.5 * sum1);
            sum2 = sum2 + sum1 * weight(mix_ind);
        end,
        if (sum2 < 1e-100)
            sum2 = 1e-100;
        end,
        dis.local_prob{ref_ind}(state_ind) = log(sum2); % Logarithmische Darstellung beachten!!!
        
    end,
end,
end

function res = map2singleres(res_tot, ref_lr)
res.num_of_words = 0;
for nr=1:res_tot.num_of_words
    if strcmp(ref_lr.name{res_tot.word_ind(1,nr)}, ref_lr.silstr)
        res.num_of_words = res.num_of_words + 1;
        res.word_ind(1,res.num_of_words) = res_tot.word_ind(1,nr);
        res.time_ind(1,res.num_of_words) = res_tot.time_ind(1,nr);        
    elseif strcmp(ref_lr.name{res_tot.word_ind(1,nr)}(5), 'r')
        res.num_of_words = res.num_of_words + 1;
        res.word_ind(1,res.num_of_words) = res_tot.word_ind(1,nr);
        res.time_ind(1,res.num_of_words) = res_tot.time_ind(1,nr);
    end
end
end

function [spec] = spec_anal(sig, init)

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

end
