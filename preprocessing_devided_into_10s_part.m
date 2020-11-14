clear all; close all;clc;
load A02.mat
% load A04.mat
number_seg=floor(length(ecg)/4000);
ecgpart=zeros(4000,number_seg);
for i =1: number_seg
    ecgpart(1:4000,i)=ecg(1+4000*(i-1):4000*i);
end
save ecgpart_02.mat ecgpart 
ecgpart=[];
load A04.mat
number_seg=floor(length(ecg)/4000);
ecgpart=zeros(4000,number_seg);
for i =1: number_seg
    ecgpart(1:4000,i)=ecg(1+4000*(i-1):4000*i);
end
save ecgpart_04.mat ecgpart 
