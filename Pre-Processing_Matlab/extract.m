Tot =118;% total number of images
% Known location of square boxes
r1=538:310:3019;
r2=825:310:3306;
c1=137:307:1980;
c2=427:307:2280;
R1=repmat(r1,7); R1 = R1(:); R1=R1(1:63) ; R1=R1+25;
R2=repmat(r2,7); R2 = R2(:); R2=R2(1:63) ;R2=R2 - 30;
C1=repmat(c1.',9); C1 = C1(:,1); C1=C1+10;
C2=repmat(c2.',9); C2 = C2(:,1);C2=C2 - 50;
Ncl = 46;% total number of classes
tic
load("Removal2.mat");
R=[R '45_1' '41_2' '26_3' '26_4' '9_16' '27_16' '35_16' '19_17' '46_74' '46_32' '44_32' '15_3' '36_51' '44_1'];
for i=1:Tot
    if ~any(i==[13 15 30 34 52 115 74 32])
    disp(i)
    Im= imread("D ("+string(i)+").jpg");
    if size(Im,3) == 3 
        Im=rgb2gray(Im);
    end
    Im = medfilt2(Im);
    Im= adapthisteq(Im);
%     T =graythresh(Im);
%     Im = imbinarize(Im,T+0.1);
%     Im = bwareaopen(Im,20);
    P=cell(1,Ncl);
   
    for j=1:Ncl
      if ~any(R(:)== string(j)+"_"+string(i))
        SubImage = Im(R1(j):R2(j),C1(j):C2(j));
        SubImage = adapthisteq(SubImage);
        T =graythresh(SubImage);
    	SubImage = ~imbinarize(SubImage,T-0.05);
        SubImage = medfilt2(SubImage,[3 3]);
        SubImage = wiener2(SubImage);
        SubImage = bwareaopen(SubImage,60);
        labeledImage = logical(SubImage);
        measurements = regionprops(labeledImage, 'BoundingBox');
        measurements1 = regionprops(labeledImage, 'Centroid');
        if size(measurements,1)==1
              BoundingBox = measurements(1).BoundingBox;
               BoundingBox(1)=BoundingBox(1)-10; BoundingBox(2)=BoundingBox(2)-10;BoundingBox(3:4)=BoundingBox(3:4)+20;
               SubImage = imcrop(SubImage, BoundingBox);
        elseif size(measurements,1)==0
            SubImage = ones(size(SubImage));
              
        else
            Ba = struct2cell(measurements); Cntr = struct2cell(measurements1);
            Ba = cell2mat(Ba.'); Cntr = cell2mat(Cntr.');
            Rmv=RemoveBox(Ba,Cntr);
            Ba(Rmv,:)=[];clear Rmv
            BoundingBox(1) = min(Ba(:,1));BoundingBox(2) = min(Ba(:,2));
            BoundingBox(3) = max(Ba(:,1)+Ba(:,3)) - BoundingBox(1);BoundingBox(4) = max(Ba(:,2)+Ba(:,4)) - BoundingBox(2);
            BoundingBox(1)=BoundingBox(1); BoundingBox(2)=BoundingBox(2);BoundingBox(3:4)=BoundingBox(3:4);
            SubImage = imcrop(SubImage, BoundingBox);
        end
        
%         SubImage = imopen(SubImage,SE);
%         SubImage = imresize(SubImage,0.125,'bicubic');
%         SubImage = imresize(SubImage,[40 40]);
        SubImage = bwmorph(SubImage, 'skel', 1);
        SubImage = bwmorph(SubImage, 'diag',1);
        SubImage = bwmorph(SubImage, 'clean', inf);
        SubImage = imdilate(SubImage,[0 0 1 0 0;0 0 1 0 0;1 1 1 1 1;0 0 1 0 0;0 0 1 0 0]);
        SubImage = bwmorph(SubImage, 'skel', 1);
%         SubImage = bwareaopen(SubImage,60);
        S        = 1 / ceil(max(size(SubImage))/80);
        SubImage = imresize(SubImage,S,'nearest','Antialiasing',false);
%         SubImage = bwmorph(SubImage, 'clean', 1);
        Ni       = zeros(80,80);
        cnt1     = ceil(size(SubImage)/2);
        cnt2     = floor(size(SubImage)/2);
        Ni(40-cnt1(1)+1:40+cnt2(1),40-cnt1(2)+1:40+cnt2(2))=SubImage;

                P{j} = Ni;
%         imwrite(Ni,string(j)+'_'+string(i)+'.jpg');
 
        
   
    
      end
    end
    save(string(i)+"M.mat",'P');
    clear P SubIage Im j boundingBox measurements labeledImage boundingBox1 boundingBox2 Tem
    end
    
end
toc