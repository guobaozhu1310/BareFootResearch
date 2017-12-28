clc
close all
clear all
clear all

s_inputdir = '\\DONGBO-PC\Data\oriData\沈阳警院数据第二批\第二批_20170707';
s_imgoutdir = 'F:\pywork\database\footweight\归一化之前\沈阳警院（334）';

st_files = dir(s_inputdir);
for i = 1:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
    mkdir(s_imgoutdir,st_files(i).name);
    st_txt = dir(fullfile(s_inputdir,st_files(i).name,'*.txt'));
    st_txtdir=fullfile(s_inputdir,st_files(i).name,st_txt.name);
    st_newtxtdir=fullfile(s_imgoutdir,st_files(i).name,st_txt.name);
    copyfile(st_txtdir, st_newtxtdir);
    
    st_imgs = dir(fullfile(s_inputdir,st_files(i).name,'*.jpg'));
    for j=1:length(st_imgs)
       st_imgdir=fullfile(s_inputdir,st_files(i).name,st_imgs(j).name);
       st_newimgdir=fullfile(s_imgoutdir,st_files(i).name,st_imgs(j).name);
       copyfile(st_imgdir, st_newimgdir);
    end
end