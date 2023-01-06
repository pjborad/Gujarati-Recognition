function Rmv = RemoveBox(Ba,Cntr)
    Bas = size(Ba,1);
    idx = 1:Bas;
    out = nchoosek(idx,2);
    for Bk = 1:size(out,1)
        V1 = Cntr(out(Bk,1),:);
        V2 = Cntr(out(Bk,2),:);
        D(Bk)  = sqrt((V1(1)-V2(1))^2 + (V1(2)-V2(2))^2); 
    end
    Loc = D>110;
    O = out(Loc,:);
    Ncount = histcounts(O)>1;
    Rmv = find(Ncount==1);
end

