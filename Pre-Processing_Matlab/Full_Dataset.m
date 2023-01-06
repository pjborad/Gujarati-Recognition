Tot = 118;
Ncl = 46;
k=0;
s= 80*80 +1 ; % total pixels (10*10 or 8*8 or 16*16 etc)
Ds=zeros(Ncl*Tot,s);
load("Removal2.mat");
R=[R '45_1' '41_2' '26_3' '26_4' '9_16' '15_3' '27_16' '35_16' '19_17' '46_74' '46_32' '44_32' '36_51' '44_1'];
tic
for i=1:Tot
    if ~any(i==[13 15 30 34 52 115 74 32])
    disp(i)
    load(string(i)+"M.mat");
    for j=1:Ncl
        if ~any(R(:)== string(j)+"_"+string(i))
        Si = P{j};
        Ir = Si.';
        Ds(j+k,:) = [Ir(:).' j];  
        end
    end
    k=k+Ncl;
    end
    
    clear P Si Ir
end
L = find(Ds(:,end)==0);
Ds(L,:) = [];
Ds=[1:s;Ds];
writematrix(Ds,'GDmnist3.csv');
% writematrix(Ds,'GujaratiData.xls');
toc
    