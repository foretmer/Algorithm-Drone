function [child]=ga_cross(parent,pcross)
%-
[popSize,dim]=size(parent);
t=randperm(popSize); 
parent=parent(t,:);  %随机打乱个体位置
child=parent;

temp1=ones(dim,1);
for i=1:popSize/2
    if rand()> pcross % 1
        continue;
    end
    pos=round(rand()*(dim-1))+1;   % 随机选择交叉点
    temp1(1:pos)=child(2*i,1:pos);
    child(2*i,1:pos)=child(2*i-1,1:pos);
    child(2*i-1,1:pos)=temp1(1:pos);  
end


