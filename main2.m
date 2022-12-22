clc,clear,close all
%% Pre-image
tic
imgir=imread('IR5.bmp');
imgvis=imread('VIS5.bmp');
%imgvis=rgb2gray(imgvis);
%imgir=rgb2gray(imgir);
imgvis=im2double(imgvis);
imgir=im2double(imgir);
imgvis = imresize(imgvis,[440 440]);
imgir = imresize(imgir,[440 440]);
%% mdlatlrr
A=cell(1,2);
A{1} = imgvis;
A{2} = imgir;
Lrr_img=cell(1,2);
Sal_img=cell(1,2);
parfor i = 1:2
    [Lrr_img{i},Sal_img{i}] = MDLatLRR(A{i});
end
%% base layer 
out1 = g_Weight_Map(Lrr_img{1,1});
out2 = g_Weight_Map(Lrr_img{1,2});
BF1 = 0.5*(Lrr_img{1,1}-Lrr_img{1,2}).*(out1-out2)+0.5*(Lrr_img{1,1}+Lrr_img{1,2});
%% detail layer fusion
nLevel=4;
sigma0 = 2;
w = floor(3*sigma0);
Sal_img{1,1}{1,1}=imgaussfilt(Sal_img{1,1}{1,1});
Sal_img{1,2}{1,1}=imgaussfilt(Sal_img{1,2}{1,1});
C_0 = double(abs(Sal_img{1,1}{1,1}) < abs(Sal_img{1,2}{1,1}));
DF = C_0.*Sal_img{1,2}{1,1} + (1-C_0).*Sal_img{1,1}{1,1};
lambda = 0.01;
for i =  nLevel : -1 : 2
    w = floor(3*sigma0);
    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
    C_0 = double(abs(Sal_img{1,1}{1,i}) < abs(Sal_img{1,2}{1,i}));
    C_0 = imfilter(C_0, h, 'symmetric');
    M = C_0.*Sal_img{1,2}{1,i} + (1-C_0).*Sal_img{1,1}{1,i};
    dd = Solve_Optimal(M,Sal_img{1,1}{1,i },Sal_img{1,2}{1,i},lambda);
    DF = DF + dd;
end
%% IMAGE SHOW
fimg=DF+BF1;
toc
figure
imshow(imgvis, 'border', 'tight', 'initialmagnification', 'fit');
set (gcf, 'Position',[0,0,440,440]);
axis normal;
figure
imshow(imgir, 'border', 'tight', 'initialmagnification', 'fit');
set (gcf, 'Position',[0,0,440,440]);
axis normal;
figure
imshow(fimg, 'border', 'tight', 'initialmagnification', 'fit');
set (gcf, 'Position',[0,0,440,440]);
axis normal;
%% MEASUREMERNT
NIQE = niqe(fimg);fprintf('NIQE=%d\n',NIQE);
MSSIM1=multissim(fimg,imgir);
MSSIM2=multissim(fimg,imgvis);
MSSIM=0.5*MSSIM1+0.5*MSSIM2;
fprintf('MSSIM=%d\n',MSSIM);
PSNR=metricsPsnr(imgir,imgvis,fimg);fprintf('PSNR=%d\n',PSNR);
QCV = metricsQcv(imgir,imgvis,fimg);fprintf('QCV=%d\n',QCV);
QAB = QABF(imgir, imgvis, fimg);fprintf('QAB=%d\n',QAB);
CrossEN1 = crossentropy(fimg,imgir);
CrossEN2 = crossentropy(fimg,imgvis);
CrossEN =0.5*CrossEN1+0.5*CrossEN2;fprintf('CrossEN=%d\n',CrossEN);
mse1=mse(fimg,imgir);
mse2=mse(fimg,imgvis);
MSE=0.5*mse1+0.5*mse2;fprintf('Mse=%d\n',MSE);