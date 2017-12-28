clc
close all
clear all
clear all
man=0;women=0;man0=1;man1=0;tman0=1;tman1=0;en=0;
s_inputdir = 'F:\pywork\database\footweight\recognition\v1.2.0.0dongbo\V1.2.0.0';
s_inputdir1 = 'F:\pywork\database\footweight\recognition\v1.2.0.0use\weight227';
s_imgoutdir1 = 'F:\pywork\database\footweight\recognition\v1.2.0.0use\weight227class\train';
s_imgoutdir2 = 'F:\pywork\database\footweight\recognition\v1.2.0.0use\weight227class\test';
st_files = dir(s_inputdir);
for i =1:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
   
    person_file = dir(fullfile(s_inputdir,st_files(i).name,'*.mat')); 
    if isempty(person_file)
        en=en+1;
        continue;
    else
    load(fullfile(s_inputdir,st_files(i).name,person_file.name));
    c=ismember(st_pinfo.s_male,'Ů');
    if c(1,1)==1
      women=women+1;
      continue;
    else
        man=man+1;
        if st_pinfo.d_weight<=70
            
          new_file = dir(fullfile(s_inputdir1,st_files(i).name,'*.jpg')); 
          if man<=650
            man0=man0+1;
            for j=1:length( new_file)
              st_imgdir=fullfile(s_inputdir1,st_files(i).name,new_file(j).name);
              st_newimgdir=fullfile(s_imgoutdir1,num2str(0),new_file(j).name);
              copyfile(st_imgdir, st_newimgdir);
            end
          else
              tman0=tman0+1;
           for j=1:length( new_file)
              st_imgdir=fullfile(s_inputdir1,st_files(i).name,new_file(j).name);
              st_newimgdir=fullfile(s_imgoutdir2,num2str(0),new_file(j).name);
              copyfile(st_imgdir, st_newimgdir);
           end
          end
        elseif st_pinfo.d_weight>=75
           
          new_file = dir(fullfile(s_inputdir1,st_files(i).name,'*.jpg')); 
          if man<=650 
            man1=man1+1;
            for j=1:length( new_file)
              st_imgdir=fullfile(s_inputdir1,st_files(i).name,new_file(j).name);
              st_newimgdir=fullfile(s_imgoutdir1,num2str(1),new_file(j).name);
              copyfile(st_imgdir, st_newimgdir);
            end
          else
              tman1=tman1+1;
             for j=1:length( new_file)
              st_imgdir=fullfile(s_inputdir1,st_files(i).name,new_file(j).name);
              st_newimgdir=fullfile(s_imgoutdir2,num2str(1),new_file(j).name);
              copyfile(st_imgdir, st_newimgdir);
             end
          end
        end
    end
              
    end          
        
end
