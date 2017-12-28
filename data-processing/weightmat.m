clc
clear
load('F:\pywork\database\footweight\归一化之前\V1.0.0.2（429）\所有信息列表.mat')
source_dir='F:\pywork\database\footweight\归一化之后\V1.0.0.2（429）\1\';   %原始数据库路径
output_dir='F:\pywork\database\footweight\归一化之后\V1.0.0.2（429）分类\';
%生成Train和Test文件夹
%for j=1:2
   Te1=[output_dir,'Test\0\'];
   mkdir(Te1)
   Te2=[output_dir,'Test\1\'];
   mkdir(Te2)
  
   Tr1=[output_dir,'Train\0\'];
   mkdir(Tr1)
   Tr2=[output_dir,'Train\1\'];
   mkdir(Tr2)
  
   Va1=[output_dir,'Val\0\'];
   mkdir(Va1)
   Va2=[output_dir,'Val\1\'];
   mkdir(Va2)
   
%end
count70=0;count75=0;count70train=0;count70test=0;count75train=0;count75test=0;
for i=1:length(st_allinfo)
    %将列表中女性样本和空样本剔除
    c=ismember(st_allinfo(i).s_male,'女');
    if isempty(st_allinfo(i).d_weight)
      st_allinfo(i).d_weight = 0;
    elseif c(1,1)==1
      st_allinfo(i).d_weight = 0; 
    elseif st_allinfo(i).n_imgcount == 0
      st_allinfo(i).d_weight = 0; 
    end
end

for  i=1:length(st_allinfo)  
    if st_allinfo(i).d_weight>0
          if st_allinfo(i).d_weight<=70
             count70=count70+1;
             p_id=st_allinfo(i).s_id;
             person_file=dir([source_dir,p_id,'\*jpg']);
             a=rand;
             if a>0.3
                count70train=count70train+1; 
                if length(person_file)>25
                   lp=25;
                else
                   lp=length(person_file);
                end
                for j=1:lp
                   img_path=[source_dir,st_allinfo(i).s_id,'\',person_file(j).name];
                   img=imread(img_path);
                   imwrite(img,[output_dir,'Train\0\',person_file(j).name]);
                end

             else
                 count70test=count70test+1;
                 if length(person_file)>25
                   lp=25;
                else
                   lp=length(person_file);
                end
                for j=1:lp
                   img_path=[source_dir,st_allinfo(i).s_id,'\',person_file(j).name];
                   img=imread(img_path);
                   imwrite(img,[output_dir,'Test\0\',person_file(j).name]);
                end
             end
          elseif st_allinfo(i).d_weight>=75
             count75=count75+1;
             p_id=st_allinfo(i).s_id;
             person_file=dir([source_dir,p_id,'\*jpg']);
             a=rand;
             if a>0.3
                 count75train=count75train+1;
                if length(person_file)>25
                   lp=25;
                else
                   lp=length(person_file);
                end
                for j=1:lp
                   img_path=[source_dir,st_allinfo(i).s_id,'\',person_file(j).name];
                   img=imread(img_path);
                   imwrite(img,[output_dir,'Train\1\',person_file(j).name]);
                end

             else
                 count75test=count75test+1;
                 if length(person_file)>25
                   lp=25;
                else
                   lp=length(person_file);
                end
                for j=1:lp
                   img_path=[source_dir,st_allinfo(i).s_id,'\',person_file(j).name];
                   img=imread(img_path);
                   imwrite(img,[output_dir,'Test\1\',person_file(j).name]);
                end
             end

        end
    end
    %db_showprocess(i,length(st_allinfo));
end
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




 