function mtx=calchomography(P,Q)
%% calculate homography from 4 pairs of point matches

if size(P,2)<4 | size(Q,2)<4
  error('Need at least 4 pairs of points for homography');
end

A=[];

for pt=1:size(P,2)
  x=P(1,pt);
  y=P(2,pt);
  u=Q(1,pt);
  v=Q(2,pt);

  A=[A ; ...
      -x -y -1 0 0 0 x*u y*u u ; ...
      0 0 0 -x -y -1 x*v y*v v];

end

[U S V]=svd(A);

mtx=reshape(V(1:9,end),3,3)';

mtx=mtx./mtx(3,3);