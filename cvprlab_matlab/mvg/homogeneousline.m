function [p1 p2]=homogeneousline(l)

% Adapted from  http://www.csse.uwa.edu.au/~pk/research/matlabfns/Projective/hline.m

l=l./l(3);

  if abs(l(1)) > abs(l(2))   % line is more vertical
    ylim = get(get(gcf,'CurrentAxes'),'Ylim');
    p1 = cross(l, [0 1 0]');
    p1=p1./p1(3);
    p2 = cross(l, [0 -1/ylim(2) 1]');
    p2=p2./p2(3);
  else                       % line more horizontal
    xlim = get(get(gcf,'CurrentAxes'),'Xlim');
    p1 = cross(l, [1 0 0]');
    p1=p1./p1(3);
    p2 = cross(l, [-1/xlim(2) 0 1]');
    p2=p2./p2(3);
  end

