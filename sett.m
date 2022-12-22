function[nFrames,lambda,L0,S0,epsilon]=sett(M)
nFrames     = size(M,2);
lambda      = 1/sqrt(max(size(M,1),size(M,2)));
L0          = repmat(median(M,2), 1, nFrames);
S0          = M - L0;
epsilon     = 5e-3*norm(M,'fro'); % tolerance for fidelity to data
end