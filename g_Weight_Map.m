function out = g_Weight_Map(I)
%% parameter setting
imgg=imgradient(I);
[m,n]=size(imgg);
Sal_Tab = zeros(m,n);
imggmean=mean(mean(imgg));
%% weight map
for i=1:m
    for j=1:n
        Sal_Tab(i,j) =abs(imgg(i,j)-imggmean);
    end
end
%% regulation
wmax=max(max(Sal_Tab));
wmin=min(min(Sal_Tab));
Sal_Tab=Sal_Tab-wmin/(wmax-wmin);
out=Sal_Tab;
end