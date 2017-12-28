clc
close all
clear all
clear all

s_inputdir = 'F:\pywork\database\footweight\recognition\v1.2.0.0use\216,102\xunlian1_216,102\v1.2.0.0_216,102';

s_imgoutdir1 = 'F:\pywork\database\footweight\recognition\v1.2.0.0use\216,102\xunlian1_216,102\train';
st_files = dir(s_inputdir);
for i =1:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
    mkdir(s_imgoutdir1,st_files(i).name);
    person_file = dir(fullfile(s_inputdir,st_files(i).name,'*.jpg'));
    if length(person_file)>20
        lp=20;
    else
        lp=length(person_file);
    end
    N=lp;           %需要抽取的图片的数量  
    num=length(person_file);       %图片的总数量  
    p=randperm(num);%随机生成1~num个随机整数  
    a=p(num-N+1:num);         %取p的前N个数  
    for j=1:N
        st_imgdir=fullfile(s_inputdir,st_files(i).name,person_file(a(j)).name);
        st_newimgdir=fullfile(s_imgoutdir1,st_files(i).name,person_file(a(j)).name);
        movefile(st_imgdir, st_newimgdir);
    end 
        
end
