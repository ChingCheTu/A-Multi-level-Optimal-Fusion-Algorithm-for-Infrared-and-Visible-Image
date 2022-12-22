clc,clear,close all
%% Pre-image
tic
fixed=imread('IR1.bmp');
img1=imread('VIS1.bmp');
fixedro=fliplr(fixed);
img1gray=rgb2gray(img1);
img1resize=imresize(img1gray, 0.3,'nearest');
%% image registation
[MOVINGREG] = registerImages(img1resize,fixedro);
%% IMCROP
targetSize = [450 450];
img1r = centerCropWindow2d(size(MOVINGREG.RegisteredImage),targetSize);
img1j = imcrop(MOVINGREG.RegisteredImage,img1r);
fixedr= centerCropWindow2d(size(fixedro),targetSize);
fixedj = imcrop(fixedro,fixedr);
imgir=im2double(fixedj);
imgvis=im2double(img1j);
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
sigma0=2;
lambda = 0.01;
w = floor(3*sigma0);
Sal_img{1,1}{1,1}=imgaussfilt(Sal_img{1,1}{1,1});
Sal_img{1,2}{1,1}=imgaussfilt(Sal_img{1,2}{1,1});
C_0 = double(abs(Sal_img{1,1}{1,1}) < abs(Sal_img{1,2}{1,1}));
DF = C_0.*Sal_img{1,2}{1,1} + (1-C_0).*Sal_img{1,1}{1,1};
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
%%
toc
figure
imshow(imgvis, 'border', 'tight', 'initialmagnification', 'fit');
set (gcf, 'Position',[0,0,450,450]);
axis normal;
figure
imshow(imgir, 'border', 'tight', 'initialmagnification', 'fit');
set (gcf, 'Position',[0,0,450,450]);
axis normal;
figure
imshow(fimg, 'border', 'tight', 'initialmagnification', 'fit');
set (gcf, 'Position',[0,0,450,450]);
axis normal;
%% MEASUREMERNT
MSSIM1=multissim(fimg,imgir);
MSSIM2=multissim(fimg,imgvis);
MSSIM=0.5*MSSIM1+0.5*MSSIM2;fprintf('MSSIM=%d\n',MSSIM);
PSNR=metricsPsnr(imgir,imgvis,fimg);fprintf('PSNR=%d\n',PSNR);
QAB = QABF(imgir, imgvis, fimg);fprintf('QAB=%d\n',QAB);
CrossEN = crossentropy(fimg,imgir);fprintf('CrossEN=%d\n',CrossEN);
mse1=mse(fimg,imgir);
mse2=mse(fimg,imgvis);
MSE=0.5*mse1+0.5*mse2;fprintf('Mse=%d\n',MSE);
