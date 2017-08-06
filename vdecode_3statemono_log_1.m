% Viterbi Decoder
% 3 hidden states Monophone
clc
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Init Params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dictionary
Dict_mono = {'_','ai','n','s','_'};
Dict = {'_','ai','n','s','_'};
% Transistion Matrix
A = [  0      1      0     0     0;
       0     0.5   0.5     0     0;
       0      0    0.4   0.6     0;
       0      0      0   0.7   0.3;
       0      0      0     0     0];
% Emission Matrix   
B = [  0    0.1    0.2     0     0;
     0.1    0.1    0.1   0.7   0.9;
     0.2    0.5    0.6   0.1   0.1;
     0.7    0.2    0.1   0.1     0;
       0    0.1      0   0.1     0];
% Take log
A = log10(A);
B = log10(B);
A(A==-Inf) = 0;
B(B==-Inf) = 0;
% Trellis Matrix
V = zeros(5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Decoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time = 1
V(4,1) = B(4,1);  %P2(0)
% time = 2
V(3,2) = B(3,2) + A(2,3) + V(4,1);  %P3(1)
V(4,2) = B(4,2) + A(2,2) + V(4,1);  %P2(1)
% time = 3
V(2,3) = B(2,3) + A(3,4) + V(3,2);                      %P4(2)
V(3,3) = B(3,3) + max(A(2,3) + V(4,2), A(3,3) + V(3,2))%P3(2)
V(4,3) = B(4,3) + A(2,2) + V(4,2);                      %P2(2)
% time = 4
V(2,4) = B(2,4) + max(A(3,4) + V(3,3), A(4,4) + V(2,3));
V(3,4) = B(3,4) + max(A(2,3) + V(4,3), A(3,3) + V(3,3));  
% time = 5
V(2,5) = B(2,5) + max(A(3,4) + V(3,4), A(4,4) + V(2,4));
% result: Trellis Matrix
Trellis_Matrix = V  
% backtracking: find optimal path
V_min = min(min(V));
V(V==0) = V_min-100;
[V_opt,V_opt_idx] = max(V);
V_len = length(V_opt);
V_max = V_opt(V_len);
% translate sequence into phonemes
Dict2 = rot90(Dict,2);
seq_phon = Dict;
for i = 1:V_len
    seq = Dict2(V_opt_idx(i));
    seq_phon(i) = seq;
end
% get sequence
seq_phon
% remove duplicates
seq_phon_final = unique(seq_phon);
disp('Sequence Decoded:')
disp([seq_phon_final,',Prob:',V_max])
   