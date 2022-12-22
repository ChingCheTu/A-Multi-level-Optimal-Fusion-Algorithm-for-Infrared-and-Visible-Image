function res=metricMI(im1,im2,fim,sw)

% function res=metricMI(im1,im2,fim,sw)
%
% This function implements the revised mutual information algorithms for fusion metric.
% im1, im2 -- input images;
% fim      -- fused image;
% sw       -- 1: revised MI; 2: Tsallis entropy (Cvejie); 3: Nava.
% res      -- metric value;
%
% IMPORTANT: The size of the images need to be 2X. This is not an
% implementation of Qu's algorithm. See the function for details.
%
% Z. Liu [July 2009]
%

% Ref: Comments on "Information measure for performance of image fusion"
% By M. Hossny et al.
% Electronics Letters Vol. 44, No.18, 2008
%
% Ref: Mutual information impoves image fusion quality assessments
% By Rodrigo Nava et al.
% SPIE Newsroom
%
% Ref: Image fusion metric based on mutual information and Tsallis entropy
% By N. Cvejie et al.
% Electronics Letters, Vol.42, No. 11, 2006

%% pre-processing
im1=normalize1(im1);
im2=normalize1(im2);
fim=normalize1(fim);

if nargin==3
    sw=1;
end

switch sw
    case 1
        % revised MI algorithm (Hossny)
        [I_fx,H_xf,H_x,H_f1]=mutual_info(im1,fim);
        [I_fy,H_yf,H_y,H_f2]=mutual_info(im2,fim);
        
        MI=2*(I_fx/(H_f1+H_x)+I_fy/(H_f2+H_y));
        res=MI;
    case 2
        q=1.85;    % Cvejic's constant
        I_fx=tsallis(im1,fim,q);
        I_fy=tsallis(im2,fim,q);
        res=I_fx+I_fy;
        
    case 3
        % MI and Tsallis entropy
        % set up constant q
        q=0.43137; % Nava's constant
        
        I_fx=tsallis(im1,fim,q);
        I_fy=tsallis(im2,fim,q);
        I_xy=tsallis(im1,im2,q);        
       
        [M_xy,H_xy,H_x,H_y]=mutual_info(im1,im2);

        MI=(I_fx+I_fy)/(H_x.^q+H_y.^q+I_xy);
        res=MI;
    otherwise
        error('Your input is wrong! Please check help file.');
end
function RES=normalize1(data)

% function RES=normalize1(data)
%
% This function is to NORMALIZE the data. 
% The data will be in the interval 0-255 (gray level) and pixel value has
% been rounded to an integer.
% 
% See also: normalize.m 
%
% Z. Liu @NRCC (Aug 24, 2009)

data=double(data);
da=max(data(:));
xiao=min(data(:));
if (da==0 & xiao==0)
    RES=data;
else
    newdata=(data-xiao)/(da-xiao);
    RES=round(newdata*255);
end

function [MI,H_xy,H_x,H_y]=mutual_info(im1,im2)

% function [MI,H_xy,H_x,H_y]=mutual_info(im1,im2)
% 
% This function is caculate the mutual information of two input images.
% im1   -- input image one;
% im2   -- input image two;
%
% MI    -- mutual information;
%
%
% Note: The input images need to be in the range of 0-255. (see function:
% normalize1.m)
%
% Z. Liu @ NRCC [July 17, 2009]

im1=double(im1);
im2=double(im2);

[hang,lie]=size(im1);
count=hang*lie;
N=256;

%% caculate the joint histogram
h=zeros(N,N);

for i=1:hang
    for j=1:lie
        % in this case im1->x (row), im2->y (column)
        h(im1(i,j)+1,im2(i,j)+1)=h(im1(i,j)+1,im2(i,j)+1)+1;
    end
end

%% marginal histogram

% this operation converts histogram to probability
%h=h./count;
h=h./sum(h(:));

im1_marg=sum(h);    % sum each column for im1
im2_marg=sum(h');   % sum each row for im2

H_x=0;   % for im1
H_y=0;   % for im2

%for i=1:count
%for i=1:N
%    if (im1_marg(i)>eps)
%        % entropy for image1
%        H_x=H_x+(-im1_marg(i)*(log2(im1_marg(i))));
%    end
%    if (im2_marg(i)>eps)
%        % entropy for image2
%        H_y=H_y+(-im2_marg(i)*(log2(im2_marg(i))));
%    end
%end

H_x=-sum(im1_marg.*log2(im1_marg+(im1_marg==0)));
H_y=-sum(im2_marg.*log2(im2_marg+(im2_marg==0)));


% joint entropy
H_xy=-sum(sum(h.*log2(h+(h==0))));

% mutual information
MI=H_x+H_y-H_xy;