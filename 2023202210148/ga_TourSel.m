function [parent,fit_parent]=ga_TourSel(bb0,popSize,combineeee,a1,cnum)
%-
tt=randperm(popSize*2);
combineeee01=combineeee(tt,:); %随机打乱个体位置
fit = zeros(1,popSize*2);
fit_parent = zeros(1,popSize);

[fitness1] = ga_fitness(bb0,2*popSize,combineeee01,a1,cnum); %调用适应度值函数

fit=fitness1;
parent = zeros(popSize,size(combineeee01,2));    % 预先分配大小
for i=1:popSize
    if fit(2*i-1) < fit(2*i)
        parent(i,:)=combineeee01(2*i-1,:);
        fit_parent(1,i) = fit(2*i-1);
    else
        parent(i,:)=combineeee01(2*i,:);
        fit_parent(1,i) = fit(2*i);
    end
end