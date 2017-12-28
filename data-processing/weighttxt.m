clc
close all
clear all
clear all
a=0;b=0;man=0;women=0;
s_inputdir = 'F:\pywork\database\footweight\归一化之前\辽警（151）';

s_imgoutdir1 = 'F:\pywork\database\footweight\归一化之后\辽警（151）db分类';

n_setw=227;
st_files = dir(s_inputdir);
for i = 4:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
    
    st_txt = dir(fullfile(s_inputdir,st_files(i).name,'*.txt'));
    st_txtdir=fullfile(s_inputdir,st_files(i).name,st_txt.name);
    fidin=fopen(st_txtdir); 
    for j=1:11
        tline=fgetl(fidin);
        if j==5
            st_filemale=tline;
        end
        if j==11
            st_fileweight=tline;
        end
    end
    male=ismember(st_filemale,'男');
    fileweight=str2num(st_fileweight);
    weightall(i,1)=fileweight;
    if male(1,1)==1
        man=man+1;
        if fileweight<=70
            a=a+1;
           %a1=rand
           weight_txt = dir(fullfile(s_inputdir,st_files(i).name,'*.jpg'));
           for k = 3:length( weight_txt)
              img = imread(fullfile(s_inputdir,st_files(i).name,weight_txt(k).name));
              new_name=[num2str(i),weight_txt(k).name];
              [h,w,d] = size(img);
              d_scale = n_setw/h;
              img = imresize(img,d_scale);
              if ndims(img)~=3
                  img = cat(3,img,img,img);
              end
              g_out = zeros(n_setw,n_setw,3);
              if size(img,1)>n_setw
                  g_out(:,1:size(img,2),:) = double(img(1:n_setw,:,:));
              else
                  g_out(1:size(img,1),1:min(n_setw,size(img,2)),:) = double(img(:,1:min(n_setw,size(img,2)),:));
              end
                   
              imwrite(uint8(g_out),fullfile(s_imgoutdir1,'0',new_name));

           end
        elseif fileweight>=75
            b=b+1;
           %a2=rand
           weight_txt = dir(fullfile(s_inputdir,st_files(i).name,'*.jpg'));
           for k = 1:length( weight_txt)
              img = imread(fullfile(s_inputdir,st_files(i).name,weight_txt(k).name));
              new_name=[num2str(i),weight_txt(k).name];
              [h,w,d] = size(img);
              d_scale = n_setw/h;
              img = imresize(img,d_scale);
              if ndims(img)~=3
                  img = cat(3,img,img,img);
              end
              g_out = zeros(n_setw,n_setw,3);
              if size(img,1)>n_setw
                  g_out(:,1:size(img,2),:) = double(img(1:n_setw,:,:));
              else
                  g_out(1:size(img,1),1:min(n_setw,size(img,2)),:) = double(img(:,1:min(n_setw,size(img,2)),:));
              end
                   
              imwrite(uint8(g_out),fullfile(s_imgoutdir1,'1',new_name));

           
           end
        end
    
    else
        women=women+1;
    end
    %db_showprocess(i,length(st_files));        
end

