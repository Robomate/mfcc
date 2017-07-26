function refw = create_word_hmms_from_phonem_eri()

addpath('/packages/speech_recognition/matlab');
lex_file = '/data/praktikum_dsv/matlab/speechdat_all.lex';
[ff, str]  = system(['wc -l ' lex_file]);
if ff
    fprintf(1, 'ERROR reading lexicon!\n');
end
[~, nrstr] = strtok(str, ' ');
nrentries = str2num(nrstr);
lexword = cell(1, nrentries);
lexphonem = cell(1, nrentries);
[lexword, lexphonem, err] = load_lexicon(lex_file);
if err
    fprintf(1, 'ERROR reading lexicon!\n');
end

word_file = '/data/eri_german/lists/commands.list';
fp = fopen(word_file, 'r');
linestr = fgetl(fp);
ii = 1;
while (linestr ~= -1)
    words{ii} = lower(linestr);
    ii = ii + 1;
    linestr = fgetl(fp);
end
words{end+1} = 'sil';
sflag = 0;
phonems = cell(1,length(words)); 
for ii=1:length(words)
    ind = strmatch(words{ii}, lexword, 'exact');
    if isempty(ind)
        fprintf(1, 'ERROR: could not find %s in lexicon!\n', words{ii});
        sflag = 1;
    else
        phonems{ii} = lexphonem{ind};
    end
end
if sflag
    fprintf(1, 'Enter missing entries in lexicon first!!!\n');
    return
end
% complete list of phonems
%  @  a:  aI  ar  b  d   E   f  h   I  k  m  N   O   Oe  OY  r  S    sp  u   U  x   Y  Z
%  a  ae  an  aU  C  e:  E:  g  i:  j  l  n  o:  oe  On  p   s  sil  t   u:  v  y:  z

hmmlist = '/data/rvg_new/config_files/monophon_hghnr39.hmm';
refph = load_hmms_covar(hmmlist, 'sil');
% loading macro file containing triphones
%refph = load_hmms_macro('/data/rvg_new/hmm/hghnr39_triphon/newMacros_8mix');
%refph = load_hmms_macro('/data/rvg_new/hmm/hghnr39_triphon/newMacros_8mix_sp_singlestate');
%refph.silstr = 'sil';

refw = merge_hmms(refph, words, phonems);
reftab_ind = zeros(1, refph.no_of_refs);
reftab_ind(1) = 0;
for ii=2:refph.no_of_refs
    reftab_ind(ii) = reftab_ind(ii-1) + refph.numstates(ii-1);
end
refw.phonindex = cell(1, length(words));
for ii=1:length(words)
    str = textscan(phonems{ii}, '%s');
    refw.phonindex{ii} = [];
    for kk=1:length(str{1})
            num = strmatch(str{1}(kk), refph.name, 'exact');
            no_states = refph.numstates(1,num);
            refw.phonindex{ii} = [refw.phonindex{ii} reftab_ind(num)+1:reftab_ind(num)+no_states];
    end
end

return;

% @   ae  ar  C   E   g   I  l  N   oe  OY  s    sp  u:  x   z
% a   aI  aU  d   E:  h   j  m  o:  Oe  p   S    t   U   y:  Z
% a:  an  b   e:  f   i:  k  n  O   On  r   sil  u   v   Y
% monophones
 reftab_ind = zeros(1, refph.no_of_refs); 
 reftab_ind(1) = 0;
 for ii=2:refph.no_of_refs
     reftab_ind(ii) = reftab_ind(ii-1) + refph.numstates(ii-1);
 end
 phonindex = cell(1, length(words));
 for ii=1:length(words)
     str = textscan(phonems{ii}, '%s');
     phonindex{ii} = [];
     for kk=1:length(str{1})
             num = strmatch(str{1}(kk), refph.name, 'exact');
             no_states = refph.numstates(1,num);
             phonindex{ii} = [phonindex{ii} reftab_ind(num)+1:reftab_ind(num)+no_states];
     end
 end
save('/data/eri_german/config_files/nn_mono_out_index.mat', 'words', 'phonindex');

% triphones
%for ii=1:length(words)
%    str = textscan(phonems{ii}, '%s');
%    phonindex{ii} = [];
%    for kk=1:length(str{1})
%            num = strmatch(str{1}(kk), refph.name, 'exact');
%            no_states = refph.numstates(1,num);
%            phonindex{ii} = [phonindex{ii} ref.state_indices{num}];
%    end
%end
%save('/data/rvg_new/nn_output/nn_tri_out_index.mat', 'words', 'phonindex');
%save('/data/rvg_new/nn_output/nn_tri_out_index_sp.mat', 'words', 'phonindex');

end

function [word, phonem, err] = load_lexicon(lex_file)

    fid_lex = fopen(lex_file, 'r');
    err = 0;
    if fid_lex == -1
        fprintf('cannot open lexicon file %s !\n', lex_file);
        err = 1;
        return
    end
    count = 0;
    linestr = fgetl(fid_lex);
    while (linestr ~= -1)
        count = count + 1;
        [word{count}, phonem{count}] = separate_word_phonems(linestr);        
        linestr = fgetl(fid_lex); 
    end
    fclose(fid_lex);
end

function [word, phonems] = separate_word_phonems(str)

        [word, phonems] = strtok(str, ' ');
        word = lower(word);
        phonems = strtrim(phonems);
        phonems = regexprep(phonems, '{|}', '');
        phonems = regexprep(phonems, '[', '');
        phonems = regexprep(phonems, ']', '');
        %phonems = strsplit(phonems);
end
%
