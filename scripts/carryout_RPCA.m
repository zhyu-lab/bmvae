function carryout_RPCA(datafile, outputfile)

tic
D=csvread(datafile,1,1);


[m,n]=size(D); % In SNV data, 3 represents missing in general.
%                           In CNV data, the missing copy number at the particular locations can be replaced by a number that does not appear in the CNV matrix.  For example, the missing locations can be filled with -1 in D.
ms=3; % ms represents missing data. In SNV data, if 3 represents missing, then ms=3; In CNV data, if -1 represents missing, then ms=-1.
omega=find(D~=ms); 
omegaC=find(D==ms);
lambda=1/sqrt(max(m,n))*(1+3*length(omegaC)/(m*n));
[A1,E1]= extendedRPCA(D,omega,lambda); % the extended RPCA model

AA1=int8(A1);

fid = fopen(outputfile, 'w');
[m,n] = size(AA1);
for i = 1:m
    for j = 1:n-1
        fprintf(fid,'%d ', AA1(i,j));
    end
    fprintf(fid,'%d\n', AA1(i,end));
end
fclose(fid);

toc

end