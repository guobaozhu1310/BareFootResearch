clc
close all
clear all
clear all

s_inputdir = 'F:\pywork\database\footweight\recognition\dongbo\temp\V1.4.0.3\NO3\test_128,59';

s_imgoutdir2 = 'F:\pywork\database\footweight\recognition\dongbo\temp\V1.4.0.3\NO3\smalltest_128,59';
st_files = dir(s_inputdir);
for i =1:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
   
    person_file = dir(fullfile(s_inputdir,st_files(i).name,'*.jpg'));
    if length(person_file)>0
        
        if length(person_file)>5
           lp=5;
        else
           lp=length(person_file);
        end
        N=lp;           %��Ҫ��ȡ��ͼƬ������  
        num=length(person_file);       %ͼƬ��������  
        for t=1:3
           mkdir(s_imgoutdir2,num2str(t));
           temp=[s_imgoutdir2,'\',num2str(t)];
           mkdir(temp,st_files(i).name);
           p=randperm(num);%�������1~num���������  
           a=p(1:N);         %ȡp��ǰN����  
           for j=1:N
              st_imgdir=fullfile(s_inputdir,st_files(i).name,person_file(a(j)).name);
              st_newimgdir=fullfile(s_imgoutdir2,num2str(t),st_files(i).name,person_file(a(j)).name);
              copyfile(st_imgdir, st_newimgdir);
           end  
        end
    end
end
