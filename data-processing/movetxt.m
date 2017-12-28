clc
close all
clear all
clear all

s_inputdir = '\\Desktop-78cpgr4\e\PROJECT\Foot_Height\data_Foot_Height\µÚ¶þÅú_20170707';
s_imgoutdir = 'G:\preprocessing\cluster-footprint\second20170707cluster-footprint Beta03';

st_files = dir(s_inputdir);
for i = 1:length(st_files)
    if strcmp(st_files(i).name,'..')||strcmp(st_files(i).name,'.')
        continue;
    end
    
    st_txt = dir(fullfile(s_inputdir,st_files(i).name,'*.txt'));
    st_txtdir=fullfile(s_inputdir,st_files(i).name,st_txt.name);
    st_newtxtdir=fullfile(s_imgoutdir,st_files(i).name,st_txt.name);
    copyfile(st_txtdir, st_newtxtdir);
end