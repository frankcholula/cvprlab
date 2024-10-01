close all;

IMG1='images/parade1.bmp';
IMG2='images/parade2.bmp';

imgshow([imread(IMG1) imread(IMG2)]);
hold on;

[hei wid C]=size(imread(IMG1));

while (1)
    [x y]=ginput (1);
    
    P = [x;y;1];
    
    Q=H*P;
    
    Q(1)=Q(1)/Q(3);
    Q(2)=Q(2)/Q(3);
   
    plot (P(1),P(2),'gx');
    plot (Q(1)+wid,Q(2),'rx');
    plot ([P(1) Q(1)+wid],[P(2) Q(2)],'b-');
    
end