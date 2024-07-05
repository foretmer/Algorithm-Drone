function F=func1(D,s)
DistanV=0;
n=size(s,2);
for i=1:(n-1)
    DistanV=DistanV+D(s(i),s(i+1));
end
DistanV=DistanV+D(s(n),s(1));
F=DistanV;
end