clc
close all
clear all
clear all

s_inputdir = 'F:\pywork\database\footweight\recognition\dongbo\temp\V1.4.0.3\NO3\Test';
s_imgoutdir1 = 'F:\pywork\database\footweight\recognition\dongbo\temp\V1.4.0.3\NO3\test_128,59';
st_files = dir(s_inputdir);
for i =1:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
    x=str2num(st_files(i).name)-1;
    mkdir(s_imgoutdir1,num2str(x));
    person_file = dir(fullfile(s_inputdir,st_files(i).name,'*.jpg')); 
    
    for j=1:length( person_file)
        st_imgdir=fullfile(s_inputdir,st_files(i).name,person_file(j).name);
        img=imread(st_imgdir);
        img=double(img);
        [m,n]=size(img);
        img0=zeros(866,409);
        for a=1:m
            for b=1:n
                img0(a,b)=img(a,b);
            end
        end
        img0=imresize(img0,[128,59]);
        st_newimgdir=fullfile(s_imgoutdir1,num2str(x),person_file(j).name);
        imwrite(uint8(img0), st_newimgdir);
    end 
        
end

